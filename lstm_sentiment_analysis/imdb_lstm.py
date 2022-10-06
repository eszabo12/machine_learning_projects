# imdb_lstm.py
# PyTorch 1.10.0-CPU Anaconda3-2020.02  Python 3.7.6
# Windows 10/11

import numpy as np
import torch as T
device = T.device('cpu')

# -----------------------------------------------------------
def print_eval_results(res):
    print("\nResults are:")
    print(f"Accuracy: {res['acc']}")
    print(f"Precision: {res['prec']}")
    print(f"Recall: {res['rec']}")
    print(f"F1: {res['f1']}")
    print("\n")

class LSTM_Net(T.nn.Module):
  def __init__(self):
    # vocab_size = 129892
    super(LSTM_Net, self).__init__()
    self.embed = T.nn.Embedding(129892, 32)
    self.lstm = T.nn.LSTM(32, 100)
    self.fc1 = T.nn.Linear(100, 2)  # 0=neg, 1=pos
 
  def forward(self, x):
    z = self.embed(x)  # x can be arbitrary shape - not
    z = z.reshape(50, -1, 32)  # seq bat embed
    lstm_output, (h_n, c_n) = self.lstm(z)
    z = lstm_output[-1]  # or use h_n. [1,100]
    z = T.log_softmax(self.fc1(z), dim=1)  # NLLLoss()
    return z

# -----------------------------------------------------------

class IMDB_Dataset(T.utils.data.Dataset):
  # 50 token IDs then 0 or 1 label, space delimited
  def __init__(self, src_file):
    all_xy = np.loadtxt(src_file, usecols=range(0,51),
      delimiter=" ", comments="#", dtype=np.int64)
    tmp_x = all_xy[:,0:50]   # cols [0,50) = [0,49]
    tmp_y = all_xy[:,50]     # all rows, just col 50
    self.x_data = T.tensor(tmp_x, dtype=T.int64)
    self.y_data = T.tensor(tmp_y, dtype=T.int64) 

  def __len__(self):
    return len(self.x_data)

  def __getitem__(self, idx):
    tokens = self.x_data[idx]
    trgts = self.y_data[idx] 
    return (tokens, trgts)

# -----------------------------------------------------------

def accuracy(model, dataset):
  # data_x and data_y are lists of tensors
  # assumes model.eval()
  true_pos = 0
  true_neg = 0
  false_pos = 0
  false_neg = 0
  correct = 0
  wrong = 0
  ldr = T.utils.data.DataLoader(dataset,
    batch_size=1, shuffle=False)
  for (batch_idx, batch) in enumerate(ldr):
    X = batch[0]  # inputs
    Y = batch[1]  # target sentiment label
    with T.no_grad():
      output = model(X)  # log-probs
   
    idx = T.argmax(output.data)

    if Y == idx:  # predicted == target
      if idx == 1:
        true_pos +=1
      else:
        true_neg +=1
      correct +=1
    else:
      wrong += 1
      if idx == 1:
        false_pos +=1
      else:
        false_neg +=1
  acc = (correct * 1.0) / (correct + wrong)
  prec = (true_pos * 1.0) / (true_pos+false_pos)
  rec = (true_pos * 1.0) / (true_pos+false_neg)
  f1 = 2 * (prec * rec)/(prec + rec)
  return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1}


def main():
  # 0. get started
  print("\nBegin PyTorch IMDB LSTM demo ")
  print("Using only reviews with 50 or less words ")
  T.manual_seed(1)
  np.random.seed(1)

  # 1. load data 
  print("\nLoading preprocessed train and test data ")
  train_file = ".//imdb_train_50w.txt"
  train_ds = IMDB_Dataset(train_file) 

  test_file = ".//imdb_test_50w.txt"
  test_ds = IMDB_Dataset(test_file) 

  bat_size = 8
  train_ldr = T.utils.data.DataLoader(train_ds,
    batch_size=bat_size, shuffle=True, drop_last=True)
  n_train = len(train_ds)
  n_test = len(test_ds)
  print("Num train = %d Num test = %d " % (n_train, n_test))

# -----------------------------------------------------------

  # 2. create network
  net = LSTM_Net().to(device)

  # 3. train model
  learning_rate = 1.0e-3
  loss_func = T.nn.NLLLoss()  # log-softmax() activation
  optimizer = T.optim.Adam(net.parameters(), lr=learning_rate)
  max_epochs = 20
  log_interval = 4  # display progress

  print("\nbatch size = " + str(bat_size))
  print("loss func = " + str(loss_func))
  print("optimizer = Adam ")
  print("learn rate = " + str(learning_rate))
  print("max_epochs = %d " % max_epochs)

  print("\nStarting training ")
  net.train()  # set training mode
  for epoch in range(0, max_epochs):
    tot_err = 0.0  # for one epoch
    for (batch_idx, batch) in enumerate(train_ldr):
      X = T.transpose(batch[0], 0, 1)
      Y = batch[1]
      optimizer.zero_grad()
      output = net(X)

      loss_val = loss_func(output, Y) 
      tot_err += loss_val.item()
      loss_val.backward()  # compute gradients
      optimizer.step()     # update weights
  
    if epoch % log_interval == 0:
      print("epoch = %4d  |" % epoch, end="")
      print("   loss = %10.4f  |" % tot_err, end="")

      net.eval()
      res = accuracy(net, train_ds)
      print(f"Accuracy: {res['acc']}")
      net.train()
  print("Training complete")

# -----------------------------------------------------------

  # 4. evaluate model
  net.eval()
  results = accuracy(net, test_ds)
  print_eval_results(results)

  # 5. save model
  print("\nSaving trained model state")
  fn = ".//Models//imdb_model_20epoch.pt"
  T.save(net.state_dict(), fn)

  # saved_model = Net()
  # saved_model.load_state_dict(T.load(fn))
  # use saved_model to make prediction(s)

  # 6. use model
  print("\nSentiment for \"the movie was a great waste of my time\"")
  print("0 = negative, 1 = positive ")
  review = np.array([4, 20, 16, 6, 86, 425, 7, 58, 64],
    dtype=np.int64)
  padding = np.zeros(41, dtype=np.int64)
  review = np.concatenate([padding, review])
  review = T.tensor(review, dtype=T.int64).to(device)
  
  net.eval()
  with T.no_grad():
    prediction = net(review)  # log-probs
  print("raw output : ", end=""); print(prediction)
  print("pseud-probs: ", end=""); print(T.exp(prediction))

  print("\nEnd PyTorch IMDB LSTM sentiment demo")

if __name__ == "__main__":
  main()
