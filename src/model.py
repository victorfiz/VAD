import torch
from torch import nn
from src.time_logs import TimerLog

class RNN(nn.Module):
    """ VAD RNN using LSTM and CrossEntropyLoss """

    def __init__(self, input_size, hidden_size, num_layers, device, verbose=False):
        """
        Initialise RNN.

        Params:
            input_size (int): Size of input features.
            hidden_size (int): Size of hidden layers.
            num_layers (int): Number of LSTM layers.
            device (str): Computation device ('cuda' or 'cpu').
            verbose (bool): Enables verbose logging.
        """
        super(RNN, self).__init__()
        self.device = device
        self.verbose = verbose
        self._init_layers(input_size, hidden_size, num_layers)

        if verbose:
            print(f'Using {self.device} device')

    def _init_layers(self, input_size, hidden_size, num_layers):
        """ Initializes network layers. """
        self.act = nn.Tanh()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.lin1 = nn.Linear(hidden_size, 26)
        self.lin2 = nn.Linear(26, 2)
        self.softmax = nn.Softmax(dim=1)
        self.to(self.device)

    def forward(self, x):
        """
        Forward pass of the network.

        Params:
            x (tensor): Input data, shape [# frames, batch_size, frame size].

        Returns:
            softmax output of the network.
        """
        h, c = self._init_hidden(x.size(1))
        x, _ = self.rnn(x, (h, c))
        x = x.contiguous().view(x.size(0), -1)
        x = self.act(self.lin1(x))
        return self.softmax(self.lin2(x))

    def _init_hidden(self, batch_size):
        """ Initializes hidden states for LSTM. """
        h = torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size).to(self.device)
        c = torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size).to(self.device)
        return h, c

    def train_model(self, librispeech, epochs=1, lrate=0.01):
        """ Trains the model. """
        timer = TimerLog()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lrate)
        
        for epoch in range(epochs):
            self._train_epoch(librispeech, optimizer, criterion, epoch)

        if self.verbose:
            print(f"Finished training in {timer.get_elapsed()} seconds.")

    def _train_epoch(self, librispeech, optimizer, criterion, epoch):
        """ Trains the model for one epoch. """
        for i, (X, y) in enumerate(librispeech):
            optimizer.zero_grad()
            output = self(X)
            loss = criterion(output, y.view(-1).long())
            loss.backward()
            optimizer.step()
            self._log_training_progress(i, len(librispeech), loss, X)

    def _log_training_progress(self, i, total, loss, X):
        """ Logs training progress. """
        if self.verbose:
            print(f"#{i}/{total}, Loss: {loss.item():.5}")
            if i % len(X) == 0:
                print(f"Sample Output:\n{torch.argmax(output, dim=1)}")

    def test_model(self, librispeech):
        """ Tests the model and computes accuracy, FRR, and FAR. """
        total, accuracy, FRR, FAR = 0, 0, 0, 0

        for i, (X, y) in enumerate(librispeech):
            total, accuracy, FRR, FAR = self._test_data(X, y, total, accuracy, FRR, FAR)
            if self.verbose:
                self._log_test_progress(i, len(librispeech.dataset), accuracy, total)

        self._print_test_results(librispeech.name, total, accuracy, FRR, FAR)

    def _test_data(self, X, y, total, accuracy, FRR, FAR):
        """ Processes and tests a batch of data. """
        output = self(X)
        y = y.view(-1)
        for j, frame in enumerate(output):
            total += 1
            prediction = torch.argmax(frame)
            accuracy += prediction == y[j]
            FRR += (prediction == 0 and y[j] == 1)
            FAR += (prediction == 1 and y[j] == 0)
        return total, accuracy, FRR, FAR

    def _log_test_progress(self, i, total, accuracy, total_classifications):
        """ Logs test progress. """
        if self.verbose:
            print(f"#{i}/{total}, Correct classifications: {accuracy}, Total classifications: {total_classifications}")

    def _print_test_results(self, dataset_name, total, accuracy, FRR, FAR):
        """ Prints final test results. """
        accuracy_percent = accuracy / total * 100
        FRR_percent = FRR / total * 100
        FAR_percent = FAR / total * 100
        print(f"Accuracy over test dataset {dataset_name}: {accuracy_percent:.2f}%")
        print(f"False Rejection Rate (FRR): {FRR_percent:.2f}%")
        print(f"False Acceptance Rate (FAR): {FAR_percent:.2f}%")