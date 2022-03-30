import ast
import logging
from typing import Callable, Dict
import flwr as fl
import torch
from utils import *
from model import *

# todo get_on_fit_config_fn

ADDRESS = "[::]"


with open(f'settings.txt', 'r') as file_dict:
    settings = file_dict.read().replace('\n', '')
    settings = ast.literal_eval(settings)

batch_size = settings["batch_size"]
total_num_clients = settings["total_num_clients"]
client_per_round = settings["client_per_round"]
lr = settings["lr"]
lr_decay = settings["lr_decay"]
local_epochs = settings["local_epochs"]
num_rounds = settings["num_rounds"]
ADDRESS =  settings["address"]



SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False


def get_eval_fn(model, testloader, device, logger):

    def evaluate(parameters, rnd):

        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        logger.info(','.join(map(str, [rnd, "evaluate", "start", time.time_ns(), time.process_time_ns(), "", ""])))
        loss, accuracy = test_server(model, testloader, device)
        logger.info(','.join(map(str, [rnd, "evaluate", "end", time.time_ns(), time.process_time_ns(), loss, accuracy])))

        return float(loss), {"accuracy":float(accuracy)}

    return evaluate

def get_on_fit_config_fn() -> Callable[[int], Dict[str, str]]:

    def fit_config(rnd: int) -> Dict[str, str]:
        config = {
            "lr": lr*(lr_decay**(rnd-1)), # -1 beacuse rounds strat from 1, but at the frist round we don't want to decay
            "batch_size": batch_size,
            "local_epochs": local_epochs,
            "rnd": rnd
        }
        return config

    return fit_config


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)
testloader, num_examples = load_data(batch_size)
model= cifarNet().to(DEVICE)

handler = logging.FileHandler("reports/server.csv", mode='w')
logger = logging.getLogger("server")
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.info("round,operation,phase,t,p,train_loss,train_acc")


strategy = fl.server.strategy.FedAvg(
    min_fit_clients=client_per_round,
    min_available_clients=total_num_clients,
    on_fit_config_fn=get_on_fit_config_fn(),
    min_eval_clients=total_num_clients,
    eval_fn=get_eval_fn(model, testloader, DEVICE, logger),
)


fl.server.start_server(f"{ADDRESS}:8080", config={"num_rounds": num_rounds}, strategy=strategy)
