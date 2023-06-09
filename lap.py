import streamlit as st

from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
import torch, numpy
from laplace import Laplace
from matplotlib import pyplot
# User options
epochs = st.number_input('Number of epochs in optimizing prior precision', min_value=1, max_value=1000, value=100)
hessian_approximation = st.selectbox('Hessian approximations', ['full', 'kron', 'diag'])
weights_subset = st.selectbox('Subset of weights', ['all','subnetwork','last_layer'])
n_epochs=st.number_input('Number of epochs in training the model', min_value=1, max_value=1000, value=100)
lr = st.slider("Learning rate", min_value=0.0, max_value=1.0,value=0.01,step=0.001)
# Take the number of hidden layers as user input
num_hidden_layers = st.number_input('Enter the number of hidden layers:', min_value=1, value=1)

# Initialize an empty list to store the number of perceptrons in each layer
hidden_sizes = []

# Create a row of columns to stack the input boxes horizontally
cols = st.beta_columns(num_hidden_layers)

# For each hidden layer, take the number of perceptrons as user input
for i in range(num_hidden_layers):
    num_perceptrons = cols[i].number_input(f'Layer {i+1}:', min_value=1, value=1)
    hidden_sizes.append(num_perceptrons)

# Display user options
st.write(f'Number of epochs: {epochs}')
st.write(f'Hessian approximation: {hessian_approximation}')
st.write(f'Subset of weights: {weights_subset}')
numpy.random.seed(0)
# Load the diabetes dataset
X_train = (torch.rand(100) * 8).unsqueeze(-1)
noise = torch.randn_like(X_train) * 0.2
y_train =X_train**2 + 2*X_train +-3 + noise
X_test = torch.linspace(-13, 13, 500).unsqueeze(-1)
noise = torch.randn_like(X_test) * 0.2
y_test =X_test**2 + 2*X_test +-3 + noise
# Create TensorDatasets from data
train_data = TensorDataset(X_train, y_train)

# Create DataLoaders for training and test data
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader=DataLoader(TensorDataset(X_test,y_test))
class Net(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(Net, self).__init__()
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_sizes)):
            if i == 0:
                self.hidden_layers.append(nn.Linear(input_size, hidden_sizes[i]))
            else:
                self.hidden_layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        x = self.output_layer(x)
        return x


input_size = 1
#hidden_sizes = [64, 32, 16, 8]
output_size = 1

  # Create an instance of the neural network
start=False
start=st.button('Start')
if start:
    # Code to run when the button is pressed
  st.write('Starting job...')
  model = Net(input_size, hidden_sizes, output_size)
  criterion = torch.nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  chart_placeholder = st.empty()
  epochs = []
  losses = []
  for i in range(n_epochs):
    for X, y in train_loader:
      optimizer.zero_grad()
      loss = criterion(model(X), y)
      loss.backward()
      optimizer.step()
    epochs.append(i+1)
    losses.append(loss.item())
    pyplot.clf()    
    pyplot.plot(epochs, losses, label="Training Loss")
    pyplot.xlabel("Epoch")
    pyplot.ylabel("Loss")
    pyplot.legend()
    chart_placeholder.pyplot(pyplot)
  la = Laplace(model, 'regression', hessian_structure=hessian_approximation,subset_of_weights=weights_subset)
  la.fit(train_loader)
  la.optimize_prior_precision(n_steps=epochs)
  # User-specified predictive approx.
  f_mu, f_var = la(X_test)
  f_mu = f_mu.squeeze().detach().cpu().numpy()
  f_sigma = f_var.squeeze().sqrt().cpu().numpy()
  
  x = X_test.flatten().cpu().numpy()
  y=y_test.flatten().cpu().numpy()
  def plot_regression(X_train, y_train, X_test, f_test, y_std, plot=True, 
                      file_name='regression_example'):
      fig, (ax1, ax2) = pyplot.subplots(nrows=1, ncols=2, sharey=True,
                                  figsize=(4.5, 2.8))
      ax1.set_title('MAP')
      ax1.scatter(X_train.flatten(), y_train.flatten(), alpha=0.3, color='tab:orange')
      ax1.plot(X_test, f_test, color='black', label='$f_{MAP}$')
      ax1.plot(X_test,y,label='True')
      ax1.legend()

      ax2.set_title('LA')
      ax2.scatter(X_train.flatten(), y_train.flatten(), alpha=0.3, color='tab:orange')
      ax2.plot(X_test, f_test, label='$\mathbb{E}[f]$')
      ax2.plot(X_test,y,label='True')
      ax2.fill_between(X_test, f_test-y_std*2, f_test+y_std*2, 
                      alpha=0.3, color='tab:blue', label='$2\sqrt{\mathbb{V}\,[y]}$')
      ax2.legend()
      ax1.set_ylabel('$y$')
      ax1.set_xlabel('$x$')
      ax2.set_xlabel('$x$')
      pyplot.tight_layout()
      pyplot.show()
      pyplot.savefig('plot.png')
      
  plot_regression(X_train, y_train, x, f_mu, f_sigma, 
                  file_name='regression_example_online', plot=False)
  st.image('plot.png')
  pyplot.close()
  from scipy.stats import norm
  x=numpy.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99])
  def z_score(confidence_level):
      alpha = 1 - confidence_level
      z = norm.ppf(1 - alpha/2)
      return z
  z_score=numpy.vectorize(z_score)
  y=z_score(x)
  ece_op=0
  y_test_num = y_test.squeeze().detach().cpu().numpy()
  lis=[]
  for num in range(len(y)):
    ratio=0
    for entry in range(len(y_test_num)):
      if abs(y_test_num[entry]-f_mu[entry])<=y[num]*f_sigma[entry]:
        ratio+=1
    ratio=ratio/len(y_test_num)
    ece_op+=abs(ratio-x[num])
    lis.append(ratio)
  ece_op=ece_op/len(x)
  pyplot.plot(x,numpy.array(lis),label='Laplace Uncertainty')
  pyplot.plot(x,x,label='Ideal case')
  pyplot.title('reliability diagram')
  pyplot.legend()
  pyplot.show()
  pyplot.savefig('reliability.png')
  st.image('reliability.png')
  st.write('Expected calibration error: ',ece_op)
  start=not start
