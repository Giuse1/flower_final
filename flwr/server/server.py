# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Flower server."""
import ast
import concurrent.futures
import copy
import timeit
from collections import OrderedDict
from logging import DEBUG, INFO, WARNING
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.utils.prune as prune

from flwr.common import (
    Disconnect,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Reconnect,
    Scalar,
    Weights,
    weights_to_parameters,
    parameters_to_weights,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from flwr.server.strategy import FedAvg, Strategy

import logging
import time
from model import *


with open(f'settings.txt', 'r') as file_dict:
    settings = file_dict.read().replace('\n', '')
    settings = ast.literal_eval(settings)


handler = logging.FileHandler("reports0/server.csv", mode='a')
logger = logging.getLogger("server")
logger.setLevel(logging.INFO)
logger.addHandler(handler)

DEPRECATION_WARNING_EVALUATE = """
DEPRECATION WARNING: Method

    Server.evaluate(self, rnd: int) -> Optional[
        Tuple[Optional[float], EvaluateResultsAndFailures]
    ]

is deprecated and will be removed in a future release, use

    Server.evaluate_round(self, rnd: int) -> Optional[
        Tuple[Optional[float], Dict[str, Scalar], EvaluateResultsAndFailures]
    ]

instead.
"""

DEPRECATION_WARNING_EVALUATE_ROUND = """
DEPRECATION WARNING: The configured Strategy uses a deprecated aggregate_evaluate
return format:

    Strategy.aggregate_evaluate(...) -> Optional[float]

This format is deprecated and will be removed in a future release. It should use

    Strategy.aggregate_evaluate(...) -> Tuple[Optional[float], Dict[str, Scalar]]

instead.
"""

DEPRECATION_WARNING_FIT_ROUND = """
DEPRECATION WARNING: The configured Strategy uses a deprecated aggregate_fit
return format:

    Strategy.aggregate_fit(...) -> Optional[Weights]

This format is deprecated and will be removed in a future release. It should use

    Strategy.aggregate_fit(...) -> Tuple[Optional[Weights], Dict[str, Scalar]]

instead.
"""

FitResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, FitRes]],
    List[BaseException],
]
EvaluateResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, EvaluateRes]],
    List[BaseException],
]
ReconnectResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, Disconnect]],
    List[BaseException],
]


class Server:
    """Flower server."""

    def __init__(
        self, client_manager: ClientManager, strategy: Optional[Strategy] = None
    ) -> None:
        self._client_manager: ClientManager = client_manager
        self.parameters: Parameters = Parameters(
            tensors=[], tensor_type="numpy.ndarray"
        )
        self.strategy: Strategy = strategy if strategy is not None else FedAvg()
        self.max_workers: Optional[int] = None

        SEED = 0
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False

        self.model = cifarNet()
        # print(self.model.conv1.weight[0][0])

        with open(f'settings.txt', 'r') as file_dict:
            settings = file_dict.read().replace('\n', '')
            settings = ast.literal_eval(settings)
        self.percentage_to_prune = settings["percentage_to_prune"]
        self.round_pruning = settings["round_pruning"]

    def set_max_workers(self, max_workers: Optional[int]) -> None:
        """Set the max_workers used by ThreadPoolExecutor."""
        self.max_workers = max_workers

    def set_strategy(self, strategy: Strategy) -> None:
        """Replace server strategy."""
        self.strategy = strategy

    def client_manager(self) -> ClientManager:
        """Return ClientManager."""
        return self._client_manager

    # pylint: disable=too-many-locals
    def fit(self, num_rounds: int) -> History:
        """Run federated averaging for a number of rounds."""
        history = History()

        # Initialize parameters
        log(INFO, "Initializing global parameters")
        self.parameters = self._get_initial_parameters()
        log(INFO, "Evaluating initial parameters")
        res = self.strategy.evaluate(parameters=self.parameters, rnd=0)
        if res is not None:
            log(
                INFO,
                "initial parameters (loss, other metrics): %s, %s",
                res[0],
                res[1],
            )
            history.add_loss_centralized(rnd=0, loss=res[0])
            history.add_metrics_centralized(rnd=0, metrics=res[1])

        # Run federated learning for num_rounds
        log(INFO, "FL starting")
        start_time = timeit.default_timer()

        for current_round in range(1, num_rounds + 1):
            # Train model and replace previous global model
            res_fit = self.fit_round(rnd=current_round)
            if res_fit and len(res_fit) == 3:
                parameters_prime, _, _ = res_fit  # fit_metrics_aggregated
                if parameters_prime:
                    params_eval = parameters_prime
                    self.parameters = parameters_prime

            elif res_fit and len(res_fit) == 4:
                parameters_prime, params_to_tx, _, _ = res_fit  # fit_metrics_aggregated
                if parameters_prime:
                    params_eval = parameters_prime
                    self.parameters = params_to_tx

            # Evaluate model using strategy implementation
            res_cen = self.strategy.evaluate(parameters=params_eval, rnd=current_round)
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                log(
                    INFO,
                    "fit progress: (%s, %s, %s, %s)",
                    current_round,
                    loss_cen,
                    metrics_cen,
                    timeit.default_timer() - start_time,
                )
                history.add_loss_centralized(rnd=current_round, loss=loss_cen)
                history.add_metrics_centralized(rnd=current_round, metrics=metrics_cen)

            # Evaluate model on a sample of available clients
            res_fed = self.evaluate_round(rnd=current_round)
            if res_fed:
                loss_fed, evaluate_metrics_fed, _ = res_fed
                if loss_fed:
                    history.add_loss_distributed(rnd=current_round, loss=loss_fed)
                    history.add_metrics_distributed(
                        rnd=current_round, metrics=evaluate_metrics_fed
                    )

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "FL finished in %s", elapsed)
        return history

    def evaluate(
        self, rnd: int
    ) -> Optional[Tuple[Optional[float], EvaluateResultsAndFailures]]:
        """Validate current global model on a number of clients."""
        log(WARNING, DEPRECATION_WARNING_EVALUATE)
        res = self.evaluate_round(rnd)
        if res is None:
            return None
        # Deconstruct
        loss, _, results_and_failures = res
        return loss, results_and_failures

    def evaluate_round(
        self, rnd: int
    ) -> Optional[
        Tuple[Optional[float], Dict[str, Scalar], EvaluateResultsAndFailures]
    ]:
        """Validate current global model on a number of clients."""

        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_evaluate(
            rnd=rnd, parameters=self.parameters, client_manager=self._client_manager
        )
        if not client_instructions:
            log(INFO, "evaluate_round: no clients selected, cancel")
            return None
        log(
            DEBUG,
            "evaluate_round: strategy sampled %s clients (out of %s)",
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `evaluate` results from all clients participating in this round
        results, failures = evaluate_clients(
            client_instructions,
            max_workers=self.max_workers,
        )
        log(
            DEBUG,
            "evaluate_round received %s results and %s failures",
            len(results),
            len(failures),
        )


        # Aggregate the evaluation results
        aggregated_result: Union[
            Tuple[Optional[float], Dict[str, Scalar]],
            Optional[float],  # Deprecated
        ] = self.strategy.aggregate_evaluate(rnd, results, failures)

        metrics_aggregated: Dict[str, Scalar] = {}
        if aggregated_result is None:
            # Backward-compatibility, this will be removed in a future update
            log(WARNING, DEPRECATION_WARNING_EVALUATE_ROUND)
            loss_aggregated = None
        elif isinstance(aggregated_result, float):
            # Backward-compatibility, this will be removed in a future update
            log(WARNING, DEPRECATION_WARNING_EVALUATE_ROUND)
            loss_aggregated = aggregated_result
        else:
            loss_aggregated, metrics_aggregated = aggregated_result

        return loss_aggregated, metrics_aggregated, (results, failures)

    def fit_round(
        self, rnd: int
    ) -> Optional[
        Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]
    ]:
        """Perform a single round of federated averaging."""

        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_fit(
            rnd=rnd, parameters=self.parameters, client_manager=self._client_manager
        )

        if not client_instructions:
            log(INFO, "fit_round: no clients selected, cancel")
            return None
        log(
            DEBUG,
            "fit_round: strategy sampled %s clients (out of %s)",
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `fit` results from all clients participating in this round
        results, failures = fit_clients(
            client_instructions,
            max_workers=self.max_workers,
        )
        log(
            DEBUG,
            "fit_round received %s results and %s failures",
            len(results),
            len(failures),
        )

        # Aggregate training results
        logger.info(','.join(map(str, [rnd, "","aggregation", "start", time.time_ns(), time.process_time_ns(), "", ""])))

        aggregated_result: Union[
            Tuple[Optional[Parameters], Dict[str, Scalar]],
            Optional[Weights],  # Deprecated
        ] = self.strategy.aggregate_fit(rnd, results, failures)


        metrics_aggregated: Dict[str, Scalar] = {}
        if aggregated_result is None:
            # Backward-compatibility, this will be removed in a future update
            log(WARNING, DEPRECATION_WARNING_FIT_ROUND)
            parameters_aggregated = None
        elif isinstance(aggregated_result, list):
            # Backward-compatibility, this will be removed in a future update
            log(WARNING, DEPRECATION_WARNING_FIT_ROUND)
            parameters_aggregated = weights_to_parameters(aggregated_result)
        else:
            parameters_aggregated, metrics_aggregated = aggregated_result

        logger.info(','.join(map(str, [rnd, "", "aggregation", "end", time.time_ns(), time.process_time_ns(), "", ""])))

        if rnd == self.round_pruning:
            print(f"PRUNING - round {rnd}")

            logger.info(','.join(map(str, [rnd, "", "pruning", "start", time.time_ns(), time.process_time_ns(), "", ""])))

            self.ordered_keys = (self.model.state_dict().keys())

            # load aggreagated parameters in model
            parameters = parameters_to_weights(parameters_aggregated)
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)

            ########################################
            # GBLOBAL PRUNING
            parameters_to_prune = (
                (self.model.conv1, 'weight'),
                (self.model.conv2, 'weight'),
                (self.model.conv3, 'weight'),
                (self.model.fc1, 'weight'),
                (self.model.fc2, 'weight'),
            )

            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=self.percentage_to_prune,
            )

            # replace weights with mask
            state_dict = copy.deepcopy(self.model.state_dict())
            to_return_list = []
            self.mask_dict = {}
            self.mask_indices_dict = {}

            for k in self.ordered_keys:
                if "weight" in k:
                    m = state_dict[k+"_mask"].cpu().type(torch.bool).numpy()
                    self.mask_dict[k] = state_dict[k+"_mask"]
                    self.mask_indices_dict[k] = self.mask_dict[k].to_sparse()._indices()
                    to_return_list.append(np.packbits(m, axis=None))

                else:
                    to_return_list.append(state_dict[k].cpu().numpy())

            parameters_aggregated_bit = weights_to_parameters(to_return_list)
            # print(to_return_list[0][0][0])

            for k in self.ordered_keys:
                if "weight" in k:
                    name, att = k.split('.')
                    prune.remove(getattr(self.model,name), name=att)

            logger.info(','.join(map(str, [rnd, "", "pruning", "end", time.time_ns(), time.process_time_ns(), "", ""])))

            return parameters_aggregated, parameters_aggregated_bit, metrics_aggregated, (results, failures)

        elif rnd > self.round_pruning:
            # load aggreagated parameters in model
            parameters = parameters_to_weights(parameters_aggregated)
            params_dict = zip(self.ordered_keys, parameters)
            sparse_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

            no_sparse_list = []
            sparse_list = [sparse_dict[k] for k in self.ordered_keys]

            for k in self.ordered_keys:
                if "weight" in k:

                     m = torch.sparse_coo_tensor(self.mask_indices_dict[k], sparse_dict[k],
                                                             tuple(self.mask_dict[k].shape)).to_dense()

                     no_sparse_list.append(m.cpu().numpy())
                else:
                    no_sparse_list.append(sparse_dict[k].cpu().numpy())

            parameters_aggregated_sparse = weights_to_parameters(sparse_list)
            parameters_aggregated = weights_to_parameters(no_sparse_list)
            return parameters_aggregated, parameters_aggregated_sparse, metrics_aggregated, (results, failures)

        return parameters_aggregated, metrics_aggregated, (results, failures)

    def disconnect_all_clients(self) -> None:
        """Send shutdown signal to all clients."""
        all_clients = self._client_manager.all()
        clients = [all_clients[k] for k in all_clients.keys()]
        instruction = Reconnect(seconds=None)
        client_instructions = [(client_proxy, instruction) for client_proxy in clients]
        _ = reconnect_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
        )

    def _get_initial_parameters(self) -> Parameters:
        """Get initial parameters from one of the available clients."""

        # Server-side parameter initialization
        parameters: Optional[Parameters] = self.strategy.initialize_parameters(
            client_manager=self._client_manager
        )
        if parameters is not None:
            log(INFO, "Using initial parameters provided by strategy")
            return parameters

        # Get initial parameters from one of the clients
        log(INFO, "Requesting initial parameters from one random client")
        random_client = self._client_manager.sample(1)[0]
        parameters_res = random_client.get_parameters()
        log(INFO, "Received initial parameters from one random client")
        return parameters_res.parameters


def reconnect_clients(
    client_instructions: List[Tuple[ClientProxy, Reconnect]],
    max_workers: Optional[int],
) -> ReconnectResultsAndFailures:
    """Instruct clients to disconnect and never reconnect."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(reconnect_client, client_proxy, ins)
            for client_proxy, ins in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,
        )

    # Gather results
    results: List[Tuple[ClientProxy, Disconnect]] = []
    failures: List[BaseException] = []
    for future in finished_fs:
        failure = future.exception()
        if failure is not None:
            failures.append(failure)
        else:
            result = future.result()
            results.append(result)
    return results, failures


def reconnect_client(
    client: ClientProxy, reconnect: Reconnect
) -> Tuple[ClientProxy, Disconnect]:
    """Instruct client to disconnect and (optionally) reconnect later."""
    disconnect = client.reconnect(reconnect)
    return client, disconnect


def fit_clients(
    client_instructions: List[Tuple[ClientProxy, FitIns]],
    max_workers: Optional[int],
) -> FitResultsAndFailures:
    """Refine parameters concurrently on all selected clients."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(fit_client, client_proxy, ins)
            for client_proxy, ins in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,
        )

    # Gather results
    results: List[Tuple[ClientProxy, FitRes]] = []
    failures: List[BaseException] = []
    for future in finished_fs:
        failure = future.exception()
        if failure is not None:
            failures.append(failure)
        else:
            # Success case
            result = future.result()
            results.append(result)
    return results, failures


def fit_client(client: ClientProxy, ins: FitIns) -> Tuple[ClientProxy, FitRes]:
    """Refine parameters on a single client."""
    fit_res = client.fit(ins)
    return client, fit_res


def evaluate_clients(
    client_instructions: List[Tuple[ClientProxy, EvaluateIns]],
    max_workers: Optional[int],
) -> EvaluateResultsAndFailures:
    """Evaluate parameters concurrently on all selected clients."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(evaluate_client, client_proxy, ins)
            for client_proxy, ins in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,
        )

    # Gather results
    results: List[Tuple[ClientProxy, EvaluateRes]] = []
    failures: List[BaseException] = []
    for future in finished_fs:
        failure = future.exception()
        if failure is not None:
            failures.append(failure)
        else:
            # Success case
            result = future.result()
            results.append(result)
    return results, failures


def evaluate_client(
    client: ClientProxy, ins: EvaluateIns
) -> Tuple[ClientProxy, EvaluateRes]:
    """Evaluate parameters on a single client."""
    evaluate_res = client.evaluate(ins)
    return client, evaluate_res
