class LinearRegression:
    '''
    Conventions:
    i = index used to iterate training data
    j = index used to iterate features
    '''
    
    params = None # intersect is params[-1]
    __X = None
    __Y = None

    def __init__(self, X, Y):
        '''
        Note: X must be a 2-dimensional array.
        If your data is [ 1, 2, 3 ] then pass it as [ [1], [2], [3] ].
        This keeps the code clean by removing the need for hacky data re-structuring.
        '''
        import numpy as np

        self.__X = np.array(X)
        self.__Y = np.array(Y)
        self.params = np.zeros( len(self.__X[0,:]) + 1 )

        
    def Hypothesis(self, data_point):
        '''
        Calculate the predicted value for datapoint "data_point".
        data_point is a 1-dimensional vector of len(params) - 1.
        '''
        sum = self.params[-1]
        for j in range( len(self.params) - 1 ):
            sum += self.params[j] * data_point[j]
        return sum

    def Cost(self):
        '''
        Least squared residuals cost function.
        '''
        sum = 0
        for i in range(len(self.__X)):
            sum += (self.Hypothesis(self.__X[i,:]) - self.__Y[i])**2
        return sum

    def __Gradient(self, j, X, Y):
        '''
        Partial derivative of cost function (Least Squared Residuals)
        with respect to params[j]
        
        X, Y = Training data
        '''
        sum = 0
        for i in range( len(self.__X[:,0]) ):
            sum += 2 * (self.Hypothesis(self.__X[i,:]) - self.__Y[i]) * \
                (1 if j == -1 else self.__X[i,j]) # params[-1] is a special case and doesn't have a matching X.
        return sum
        
    def Iterate(self, learning_rate=0.001):
        '''
        Train the model one step at a time.
        '''
        for j in range( -1, len(self.params) - 1 ):
            change = learning_rate * self.__Gradient(j, self.__X, self.__Y)
            self.params[j] -= change

    def Train(self, learning_rate=0.001, max_iterations=1000, min_change=0.01):
        prev_cost = self.Cost()
        for _ in range(max_iterations):
            self.Iterate(learning_rate)
            if prev_cost - self.Cost() < min_change:
                return
