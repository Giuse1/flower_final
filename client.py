import ast
import copy
import optparse
import time
from utils import *
from model import *
import flwr as fl
import logging
import os
import re
import torch.nn.utils.prune as prune



list_dir = [x for x in os.listdir() if "reports" in x]
list_numbers_dirs = [int(re.findall(r'[0-9]+', x)[0]) for x in list_dir]
max_dirs = max(list_numbers_dirs)
folder=f"reports{int(max_dirs)}"

with open(f'settings.txt', 'r') as file_dict:
    settings = file_dict.read().replace('\n', '')
    settings = ast.literal_eval(settings)


batch_size = settings["batch_size"]
total_num_clients = settings["total_num_clients"]
client_per_round = settings["client_per_round"]
ADDRESS =  settings["address"]


SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

class CifarClient(fl.client.NumPyClient):

    def __init__(self, id, trainloader, testloader, batch_size) -> None:
        super(CifarClient, self).__init__()

        self.id = id
        self.batch_size = batch_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.net = cifarNet().to(self.device)
        self.starting_dict = copy.deepcopy(self.net.state_dict())
        self.trainloader = trainloader
        self.testloader = testloader
        self.num_examples = {"trainset": len(trainloader.dataset), "testset": len(testloader.dataset)}
        self.current_round = -1
        self.ordered_keys = (self.net.state_dict().keys())

        self.logger = self.setup_logger('client_logger', f'{folder}/client_{self.id}.csv')

    def get_parameters(self):

        try:
            for k in self.ordered_keys:
                if "weight" in k:
                    name, att = k.split('.')
                    prune.remove(getattr(self.net,name), name=att)
            print("GET PARAMS AFTER PRUNE REMOVE")
        except:
            pass

        to_return = [self.net.state_dict()[k].cpu().numpy() for k in self.ordered_keys]

        return to_return

    def set_parameters(self, parameters):

        print("BEFORE SET PARAMETERS")
        print(self.net.conv1.weight[0][0])

        params_dict = zip(self.ordered_keys, parameters)
        if self.current_round < (settings["round_pruning"]+1):
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.net.load_state_dict(state_dict, strict=True)

        elif self.current_round == (settings["round_pruning"]+1):
            tmp = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.net.load_state_dict(self.starting_dict, strict=True)
            self.mask_dict = {}
            for k in self.ordered_keys:
                if "weight" in k:
                    name, att = k.split('.')
                    self.mask_dict[k]=tmp[k]
                    prune.custom_from_mask(getattr(self.net,name), name=att, mask=tmp[k])

        else:
            received_state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.net.load_state_dict(received_state_dict, strict=True)

            for k in self.ordered_keys:
                if "weight" in k:
                    name, att = k.split('.')
                    prune.custom_from_mask(getattr(self.net, name), name=att, mask=self.mask_dict[k])

        # if self.current_round == (settings["round_pruning"]+1):
        #     print(f"ROUND: {self.current_round} - PRUNING")
        #     client_mask = (~torch.isnan(self.net.conv1.weight)).float()
        #     self.net.conv1.weight = nn.Parameter(torch.nan_to_num(self.net.conv1.weight, 0))
        #     prune.custom_from_mask(self.net.conv1, name='weight', mask=client_mask)

        print("AFTER SET PARAMETERS")
        print(self.net.conv1.weight[0][0])

    def fit(self, parameters, config):
        

        self.current_round = config["rnd"]
        print(f"CLIENT {self.id} TRAIN - ROUND {self.current_round}")


        self.set_parameters(parameters)

        self.logger.info(','.join(map(str, [self.current_round,"training","start",time.time_ns(),time.process_time_ns(),"",""])))
        self.train(lr=config["lr"],local_epochs=config["local_epochs"])
        self.logger.info(','.join(map(str, [self.current_round,"training","end",time.time_ns(),time.process_time_ns(),"",""])))

        self.logger.info(','.join(map(str, [self.current_round, "evaluation", "start", time.time_ns(), time.process_time_ns(), "", ""])))
        train_loss, train_acc = self.test()
        self.logger.info(','.join(map(str, [self.current_round, "evaluation", "end", time.time_ns(), time.process_time_ns(), train_loss, train_acc])))

        print("AFTER TRAINING")
        print(self.net.conv1.weight[0][0])

        return self.get_parameters(), self.num_examples["trainset"], {}

    def evaluate(self, parameters, config):
        raise Exception("ENTERED EVALUATE")

        self.current_round = config["rnd"]
        self.set_parameters(parameters)
        self.logger.info(','.join(map(str, [self.current_round,"evaluate","start",time.time_ns(),time.process_time_ns(),"",""])))
        loss, accuracy = self.test()
        self.logger.info(','.join(map(str, [self.current_round,"evaluate","end",time.time_ns(),time.process_time_ns(),"",""])))

        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}

    def train(self, lr, local_epochs):

        self.net.train()

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.net.parameters(), lr=lr)

        for _ in range(local_epochs):
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.net(images), labels)
                loss.backward()
                optimizer.step()

    def test(self):
        self.net.eval()

        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0
        with torch.no_grad():
            for data in self.testloader: # todo check
                images, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.net(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        return loss, accuracy

    def setup_logger(self, name, log_file, level=logging.INFO):

        handler = logging.FileHandler(log_file,mode='w')
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.addHandler(handler)
        logger.info("round,operation,phase,t,p,train_loss,train_acc")

        return logger

parser = optparse.OptionParser()
parser.add_option('-i', dest='id', type='int')
(options, args) = parser.parse_args()

time.sleep(options.id)
trainloader, testloader = get_cifar_iid(batch_size, total_num_clients, options.id)

fl.client.start_numpy_client(f"{ADDRESS}:8080", client=CifarClient(options.id, trainloader, testloader, batch_size))
