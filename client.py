import ast
import optparse
import time
from utils import *
from model import *
import flwr as fl
import logging

# todo 20 client,
# todo stesso test-set, quello globale


with open(f'settings.txt', 'r') as file_dict:
    settings = file_dict.read().replace('\n', '')
    settings = ast.literal_eval(settings)


batch_size = settings["batch_size"]
total_num_clients = settings["total_num_clients"]
client_per_round = settings["client_per_round"]
ADDRESS =  settings["address"]


SEED = 0
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
        self.net = cifarNet().to(self.device)
        self.trainloader = trainloader
        self.testloader = testloader
        self.num_examples = {"trainset": len(trainloader.dataset), "testset": len(testloader.dataset)}

        self.logger = self.setup_logger('client_logger', f'reports/report_{self.id}.csv')

    def get_parameters(self):
        # print(f"CLIENT {self.id}GET PARAMS")
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):

        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)


    def fit(self, parameters, config):

        current_round = config["rnd"]

        print(f"CLIENT {self.id} TRAIN - ROUND {current_round}")


        # print(f"CLIENT {self.id}FIT: BEFORE SET PARAMS")
        # print(self.net.conv1.bias.data)
        self.set_parameters(parameters)
        # print(f"CLIENT {self.id}FIT: AFTER SET PARAMS")
        # print(self.net.conv1.bias.data)

        self.logger.info(','.join(map(str, [current_round,"training","start",time.time_ns(),time.process_time_ns(),"",""])))
        self.train(lr=config["lr"],local_epochs=config["local_epochs"])

        # print(f"CLIENT {self.id}FIT: AFTER TRAIN")
        # print(self.net.conv1.bias.data)

        self.logger.info(','.join(map(str, [current_round,"training","end",time.time_ns(),time.process_time_ns(),"",""])))

        self.logger.info(','.join(map(str, [current_round, "evaluation", "start", time.time_ns(), time.process_time_ns(), "", ""])))
        train_loss, train_acc = self.test()
        self.logger.info(','.join(map(str, [current_round, "evaluation", "end", time.time_ns(), time.process_time_ns(), train_loss, train_acc])))

        return self.get_parameters(), self.num_examples["trainset"], {}

    def evaluate(self, parameters, config):
        raise Exception("ENTERED EVALUATE")

        current_round = config["rnd"]
        # print(f"CLIENT {self.id}EVALUATE: BEFORE SET PARAMS")
        # print(self.net.conv1.bias.data)
        self.set_parameters(parameters)
        # print(f"CLIENT {self.id}EVALUATE: AFTER SET PARAMS")
        # print(self.net.conv1.bias.data)
        self.logger.info(','.join(map(str, [current_round,"evaluate","start",time.time_ns(),time.process_time_ns(),"",""])))
        loss, accuracy = self.test()
        self.logger.info(','.join(map(str, [current_round,"evaluate","end",time.time_ns(),time.process_time_ns(),"",""])))

        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}

    def train(self, lr, local_epochs):

        self.net.train()
        # print(f"CLIENT {self.id}TRAIN")

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
        # print(f"CLIENT {self.id}TEST")
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

print(f"Client {options.id}")
fl.client.start_numpy_client(f"{ADDRESS}:8080", client=CifarClient(options.id, trainloader, testloader, batch_size))
