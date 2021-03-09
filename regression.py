import numpy as np
import csv
import numpy as np
from numpy.linalg import inv
import math
import random
import matplotlib.pyplot as plt


# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_dataset(filename):
    """
    TODO: implement this function.

    INPUT: 
        filename - a string representing the path to the csv file.

    RETURNS:
        An n by m+1 array, where n is # data points and m is # features.
        The labels y should be in the first column.
    """
    dataset = None
    list = []
    with open(filename, newline= '' ) as csvfile:
        reader = csv.reader(csvfile)
        # dataset = list(reader)
        i = 0
        # iterate through the file row by row
        for row in reader:
            if i == 0:
                i += 1
                continue
            row.pop(0)
            list.append(row)


    dataset = np.array(list)

    return dataset


def print_stats(dataset, col):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        col     - the index of feature to summarize on. 
                  For example, 1 refers to density.

    RETURNS:
        None
    """

    num_points = 0
    sum = 0
    for i in dataset:
        sum += float(i[col])
        num_points += 1

    mean = sum / num_points

    sum_SD = 0
    for k in dataset:
        sum_SD += (mean - float(k[col])) ** 2

    print(num_points) # print the number of elements
    print('{:.2f}'.format(mean)) # print the mean of elements
    print('{:.2f}'.format(math.sqrt(sum_SD/(num_points-1)))) # print the standard deviation of the sample
    pass


def regression(dataset, cols, betas):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]

    RETURNS:
        mse of the regression model mean square error
    """
    # check if the input betas is np array, if so, convert the copy of it to a list new_betas
    if type(betas) is np.ndarray:
        list3 = []
        for i in betas:
            list3.append(i)
        new_betas = list3
    else:

        new_betas = betas.copy()

    # seperate b0 from betas
    b0 = new_betas[0] # copy and pop out the first element of the beta, sum later
    new_betas.pop(0)

    sum = 0
    num = 0

    # model: b0 + b1x1 + b2x2 +... BmXm
    # iterate through each row of the array
    for i in dataset:
        model = b0 # initialize the model of with b0
        a = 0
        for k in new_betas:
            # add bi * xi to model
            model += float(k) * float(i[int(cols[a])]) # multiply elements of beta and x at the specified column
                                                        # int(col[a]): the int value of column at the index of k
            a += 1

        # add the value of the formula to the total sum
        sum += (model - float(i[0])) ** 2  # (mse (mean squared error) = model - y) ** 2
        num += 1

    mse = sum/(num)
    return mse


def gradient_descent(dataset, cols, betas):
    """
    TODO: implement this function.
    INPUT:
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]
    RETURNS:
        An 1D array of gradients
    """
    # check if the input betas is np array, if so, convert the copy of it to a list new_betas
    if type(betas) is np.ndarray:
        list3 = []
        for i in betas:
            list3.append(i)
        new_betas = list3
    else:

        new_betas = betas.copy()

    # sperate the betas[0] from new_betas
    b0 = new_betas[0]
    new_betas.pop(0)

    sum = 0
    num = 0

    list1 = [] # create a list of partial derivative of betas with length of betas, initialized = 0s
    for i in range(len(new_betas)):
        list1.append(0)

    # model: b0 + b1x1 + b2x2 +... BmXm
    # iterate through each row of the array
    for i in dataset:
        model = b0
        a = 0
        for k in new_betas:
            # add bi * xi to model
            model += float(k) * float(i[int(cols[a])])
            a += 1

        # calculate the value from betas[1] to betas[length - 1]
        for z in range(len(new_betas) ):
            # sum (model - y) * xi to list1z
            list1[z] += (model - float(i[0])) * float(i[int(cols[z])])


        # calculate the gradient of betas[0]
        sum += model - float(i[0]) # b0
        num += 1

    list = []
    # times each value of list with 2/num
    for z in range(len(new_betas)):
        list1[z] = list1[z] * 2/num

    list.append(sum*2/num) # the first gradient descent
    list = list + list1

    grads = np.array(list)
    return grads


def iterate_gradient(dataset, cols, betas, T, eta):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]
        T       - # iterations to run
        eta     - learning rate

    RETURNS:
        None
    """
    new_betas = betas.copy()

    # iterate for number of T times
    for i in range(T):

        list =[]
        a = new_betas.copy()
        for k in range(len(new_betas)):
            new_betas[k] = new_betas[k] - (gradient_descent(dataset, cols, a)[k])*eta # Beta.t = Beta.t-1 * eta
            list.append('{:.2f}'.format(new_betas[k]))

        print(i+1, '{:.2f}'.format(regression(dataset,cols,new_betas)), *list, sep=' ')
        # print(gradient_descent(dataset,cols,betas))
    pass


def compute_betas(dataset, cols):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.

    RETURNS:
        A tuple containing corresponding mse and several learned betas
    """
    list1 = [] # list of list of x
    y_list = [] # list of y
    for row in dataset:
        list = [] # list of x in a row
        list.append(1)
        y_list.append(float(row[0]))
        for i in cols:
            list.append(float(row[int(i)]))
        list1.append(list)
    # get the x matrix and the y matrix
    x_array = np.array(list1)
    y_array = np.array(y_list)

    x_transpose = np.transpose(x_array)

    best_params = inv(x_transpose.dot(x_array)).dot(x_transpose).dot(y_array) # (X.T * X )^-1 * X * Y (X, Y are matrices)

    betas = best_params

    mse = regression(dataset, cols, betas)

    return (mse, *betas)

def compute_betas_helper(dataset, cols):
    list1 = []
    y_list = []
    for row in dataset:
        list = []
        list.append(1)
        y_list.append(float(row[0]))
        for i in cols:
            list.append(float(row[int(i)]))
        list1.append(list)
    # get the x matrix and the y matrix
    x_array = np.array(list1)
    y_array = np.array(y_list)

    x_transpose = np.transpose(x_array)

    best_params = inv(x_transpose.dot(x_array)).dot(x_transpose).dot(y_array)

    betas = best_params
    return betas

def predict(dataset, cols, features):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        features- a list of observed values

    RETURNS:
        The predicted body fat percentage value
    """
    # check if the input betas is np array, if so, convert the copy of it to a list new_betas
    betas = compute_betas_helper(dataset, cols)

    if type(betas) is np.ndarray:
        list3 = []
        for i in betas:
            list3.append(i)
        new_betas = list3
    else:

        new_betas = betas.copy()

    # seperate b0 from betas
    b0 = new_betas[0]
    new_betas.pop(0)
    sum = 0

    # times b0 with corresponding features and sum the value
    for i in range(len(new_betas)):
        sum += float(new_betas[i]) * (float(features[i]))

    sum += b0 # add b0 to sume

    result = sum # the predict body fat based on features and
    return result


def synthetic_datasets(betas, alphas, X, sigma):
    """
    TODO: implement this function.

    Input:
        betas  - parameters of the linear model
        alphas - parameters of the quadratic model
        X      - the input array (shape is guaranteed to be (n,1))
        sigma  - standard deviation of noise

    RETURNS:
        Two datasets of shape (n,2) - linear one first, followed by quadratic.
    """
    #mu, sigma = 0, 0.1  # mean and standard deviation

    # z_i = np.random.normal(0, sigma)

    list= [] # linear dataset
    # iterate every elements in X, for linear datasets
    for k in X:
        z_i = np.random.normal(0, sigma) # random noise value

        list1 = []
        a = float(k[0]) # the value of X at k
        sum1 = 0

        for i in range(len(betas)):
            if (i == 0):
                sum1 += betas[0]
                continue
            sum1 += betas[i] * k[0]

        sum1 += z_i # col[0] = yi = Beta0 + Betai * Xi + Zi
        list1.append(sum1)
        list1.append(a)
        list.append(list1) # append the data pairs to the linear dataset list

    return_linear = np.array(list)

    qudratic_list = [] # quadratic is similar to the linear list, except formula differ

    for j in X:
        z_i = np.random.normal(0, sigma)

        list2 = []
        b = float(j[0])
        sum2 = 0
        for h in range(len(alphas)):
            if i == 0:
                sum2 += alphas[0]
                continue
            sum2 += alphas[h] * j[0] * j[0]

        sum2 += z_i
        list2.append(sum2)
        list2.append(b)
        qudratic_list.append(list2)

    return_quadratic = np.array(qudratic_list)

    return return_linear, return_quadratic


def plot_mse():
    from sys import argv
    if len(argv) == 2 and argv[1] == 'csl':
        import matplotlib
        matplotlib.use('Agg')

    # TODO: Generate datasets and plot an MSE-sigma graph
    list = []
    for i in range(1000):
        list1 = []
        a = random.uniform(-100, 100)
        list1.append(a)
        list.append(list1)
    X = np.array(list)
    betas = np.array([3, 4, 5, 6])
    alphas = np.array([1, 2, 6, 7])

    X_list_linear = []
    X_list_qua = []
    sigma_list = [1/(10 ** 4), 1/(10 ** 3), 1/(10 ** 2), 1/(10), 1, 10, 10**2, 10**3, 10**4,10**5,]


    sigma = (1/(10**4))
    linear_4, quadratic_4 = mse_helper(betas, alphas, X, sigma)
    X_list_linear.append(linear_4)
    X_list_qua.append((quadratic_4))

    sigma = (1/(10**3))
    linear_2, quadratic_2 = mse_helper(betas, alphas, X, sigma)
    X_list_linear.append(linear_2)
    X_list_qua.append((quadratic_2))

    sigma = (1 / (10 ** 2))
    linear_3, quadratic_3 = mse_helper(betas, alphas, X, sigma)
    X_list_linear.append(linear_3)
    X_list_qua.append((quadratic_3))

    sigma = (1 / (10 ))
    linear_4, quadratic_4 = mse_helper(betas, alphas, X, sigma)
    X_list_linear.append(linear_4)
    X_list_qua.append((quadratic_4))

    sigma = (1)
    linear_5, quadratic_5 = mse_helper(betas, alphas, X, sigma)
    X_list_linear.append(linear_5)
    X_list_qua.append((quadratic_5))

    sigma = (10)
    linear_6, quadratic_6 = mse_helper(betas, alphas, X, sigma)
    X_list_linear.append(linear_6)
    X_list_qua.append((quadratic_6))

    sigma = (10 **2)
    linear_7, quadratic_7 = mse_helper(betas, alphas, X, sigma)
    X_list_linear.append(linear_7)
    X_list_qua.append((quadratic_7))

    sigma = (10**3)
    linear_8, quadratic_8 = mse_helper(betas, alphas, X, sigma)
    X_list_linear.append(linear_8)
    X_list_qua.append((quadratic_8))

    sigma = (10 ** 4)
    linear_9, quadratic_9 = mse_helper(betas, alphas, X, sigma)
    X_list_linear.append(linear_9)
    X_list_qua.append((quadratic_9))

    sigma = (10 ** 5)
    linear_10, quadratic_10 = mse_helper(betas, alphas, X, sigma)
    X_list_linear.append(linear_10)
    X_list_qua.append((quadratic_10))
    #print(s1)
    f = plt.figure()
    plt.plot( sigma_list, X_list_linear,'-o', label = 'linear')
    plt.plot( sigma_list, X_list_qua, '-o', label='qua')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Sigma')
    plt.ylabel('MSE')
    plt.legend()

    f.savefig('mse.pdf')
    # plt.show()
    #print(X_list_linear)
    #print(X_list_qua)

    plt.plot(X_list_linear)
    #print(s2)
    # print(a)


def mse_helper(betas, alphas, X, sigma):
    s1, s2 = synthetic_datasets(betas, alphas, X, sigma)
    linear_1 = compute_betas(s1, [1])[0]
    quadratic_1 = compute_betas(s2, [1])[0]

    return linear_1, quadratic_1

if __name__ == '__main__':
    # get_dataset('bodyfat.csv')
    # dataset = get_dataset('bodyfat.csv')
    # print_stats(get_dataset('bodyfat.csv'), 1)
    # regression(dataset, cols=[2,3], betas=[0,0,0])
    # regression(dataset, cols=[2,3,4], betas=[0,-1.1,-.2,3])
    # print('')
    # regression(dataset, cols=[2,3,4], betas=[0,-1.1,-.2,3])
    # gradient_descent(dataset, cols=[2,3], betas=[0,0,0])
    # iterate_gradient(dataset, cols=[1,8], betas=[400,-400,300], T=10, eta=1e-4)
    # compute_betas(dataset, cols=[1, 2])
    # predict(dataset, cols=[1, 2], features=[1.0708, 23])
    #print(synthetic_datasets(np.array([0, 2]), np.array([0, 1]), np.array([[4]]), 1))
    ### DO NOT CHANGE THIS SECTION ###
    plot_mse()
