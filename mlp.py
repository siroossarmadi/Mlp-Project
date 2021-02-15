import numpy as np
import matplotlib.pyplot as plt

class MLP(object):
    
    def __init__(self, iris, epoch,  learn_rate, split, file_output):
        self.iris = iris
        self.data_split(split)
        self.learn_rate = learn_rate
        self.epoch = epoch
        self.hidden_weight =np.random.uniform(low=0, high=1, size=(5,4))
        self.output_weight = np.random.uniform(low=0, high=1, size=(3,5))
        self.hidden_bias = np.random.random(5)
        self.output_bias = np.random.random(3)
        self.error = 0
        self.hit = 0
        self.accuracy = 0
        self.all_error = []
        self.all_accuracy = []
        self.file_output= file_output
        
    def sigmoid(self, result):
        sigmoid = []
        for r in result:
            sigmoid.append(1 / (1 + np.exp(-r)))
        return sigmoid
    
    def prediction(self, sigmoid):
        self.output =[]
        for x in sigmoid:
            if x < 0.5:
                o = 0
            else:
                o = 1    
            self.output.append(o)
        return self.output
    
    def update_weights(self, iris, hidden_sigmoid, output_sigmoid):
        old_weights = self.output_weight.copy()
        for x, value in enumerate(self.output_weight):
            for y, _ in enumerate(value):
                self.output_weight[x][y]-=(output_sigmoid[x]-iris[4+x])*output_sigmoid[x]*(1-output_sigmoid[x])*hidden_sigmoid[y]*self.learn_rate
        
        for x, value in enumerate(self.hidden_weight):
            for y, _ in enumerate(value):
                sum_ = 0
                for z in range(3):
                    sum_+=(output_sigmoid[z]-iris[4+z])*output_sigmoid[z]*(1-output_sigmoid[z])*old_weights[z][x]
                self.hidden_weight[x][y]-=sum_*hidden_sigmoid[x]*(1-hidden_sigmoid[x])*iris[y]*self.learn_rate

    def update_bias(self, iris, hidden_sigmoid, output_sigmoid):
        old_weights = self.output_weight.copy()
        for x, value in enumerate(self.output_bias):
            self.output_bias[x]-=(output_sigmoid[x]-iris[4+x])*output_sigmoid[x]*(1-output_sigmoid[x])*self.learn_rate
        for x, value in enumerate(self.hidden_bias):
            sum_ = 0
            for z in range(3):
                sum_+=(output_sigmoid[z]-iris[4+z])*output_sigmoid[z]*(1-output_sigmoid[z])*old_weights[z][x]
            self.hidden_bias[x]-=sum_*hidden_sigmoid[x]*(1-hidden_sigmoid[x])*self.learn_rate
        
            
    def data_split(self, percentage):
        np.random.shuffle(self.iris)
        self.train_data = self.iris[:int((len(self.iris))*percentage/100)]
        self.test_data = self.iris[int((len(self.iris))*percentage/100):]

    def train(self, iris):
        self.error = 0
        for data in iris:
            summation =[]
            for x, layer in enumerate(self.hidden_weight):
                summation.append(np.dot(data[:4], layer)+self.hidden_bias[x])
            hidden_sigmoid = self.sigmoid(summation)
            summation.clear()
            for x, layer in enumerate(self.output_weight):
                summation.append(np.dot(hidden_sigmoid, layer)+self.output_bias[x])
            output_sigmoid = self.sigmoid(summation)
            #print(output_sigmoid)
            for key, value in enumerate(output_sigmoid):
                self.error += pow(data[(4+key)]-value,2)
            #self.update_bias(data,  hidden_sigmoid, output_sigmoid)
            self.update_weights( data, hidden_sigmoid, output_sigmoid)
        self.all_error.append(self.error/6)
        
    def test(self, iris):
        self.hit = 0
        for data in iris:
            summation =[]
            for x, layer in enumerate(self.hidden_weight):
                summation.append(np.dot(data[:4], layer)+self.hidden_bias[x])
            hidden_sigmoid = self.sigmoid(summation)
            summation.clear()
            for x, layer in enumerate(self.output_weight):
                summation.append(np.dot(hidden_sigmoid, layer)+self.output_bias[x])
            output_sigmoid = self.sigmoid(summation)
            #print(output_sigmoid)
            pred = [self.prediction(output_sigmoid)]+[data[-3:]]
            if np.all(pred[0]==pred[1]):
                self.hit +=1    
        self.all_accuracy.append(self.hit/30)
            
    def single_run(self):
        self.train(self.train_data)
        self.test(self.test_data)
    
    def plot(self, error, accuracy):
        scale_error = np.log10(error)
        plt.figure(1)
        plt.plot(scale_error, color='blue', linewidth=2, label = 'error')
        plt.plot(accuracy, color='red', linewidth=2, label = 'accuracy')
        plt.title('Learning Rate ' + str(self.learn_rate))
        plt.xlabel('Epoch')
        plt.ylabel('Quantity')
        plt.legend()
        

    def run(self):
        np.random.shuffle(self.iris) 
        for _ in range( self.epoch):
            self.cur_epoch = _
            self.single_run()
        self.plot(self.all_error,self.all_accuracy)
        export =[]
        export.append(self.all_error)
        export.append(self.all_accuracy)
        np.savetxt(self.file_output, np.transpose(export), delimiter=",",header="Error, Accuracy")