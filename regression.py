import numpy as np
from matplotlib import pyplot as plt
import csv
import numpy as np
import math


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
    print(num_points)
    print('{:.2f}'.format(mean))
    print('{:.2f}'.format(math.sqrt(sum_SD/(num_points-1))))
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
    # sperate b0 from betas
    new_betas = betas.copy()
    b0 = new_betas[0]
    new_betas.pop(0)

    sum = 0
    num = 0
    # iterate through each row of the array
    for i in dataset:
        model = b0
        a = 0
        for k in new_betas:
            # add bi * xi to model
            model += float(k) * float(i[int(cols[a])])
            a += 1

        # add the value of the formula to the total sum
        sum += (model - float(i[0])) ** 2
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
    new_betas = betas.copy()

    # sperate the betas[0] from new_betas
    b0 = new_betas[0]
    new_betas.pop(0)

    sum = 0
    num = 0

    my_vars = {}

    list1 = []
    for i in range(len(new_betas)):
        list1.append(0)


    for i in dataset:
        model = b0
        a = 0
        for k in new_betas:
            # add bi * xi to model
            model += float(k) * float(i[int(cols[a])])
            a += 1

        # add the value of the formula to the total sum


        for z in range(len(new_betas) ):

            list1[z] += (model - float(i[0])) * float(i[int(cols[z])])
            # print(z)

        sum += model - float(i[0])
#        sum2 += (model - float(i[0])) * float(i[int(cols[a-1])])
        num += 1

#    print(list1)
    list = []
    for z in range(len(new_betas)):
        list1[z] = list1[z] * 2/num

    list.append(sum*2/num)
    list = list + list1
    # print(sum2*2/num)
    b = 0

    grads = list
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

    for i in range(T):
        # print(i+1)
        # print(betas)
        list =[]
        for k in range(len(new_betas)):
            new_betas[k] = new_betas[k] - (gradient_descent(dataset, cols, new_betas)[k])*eta
            list.append(new_betas[k])

        print(i+1, regression(dataset,cols,new_betas), list)
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
    betas = None
    mse = None
    return (mse, *betas)


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
    result = None
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
    return None, None


def plot_mse():
    from sys import argv
    if len(argv) == 2 and argv[1] == 'csl':
        import matplotlib
        matplotlib.use('Agg')

    # TODO: Generate datasets and plot an MSE-sigma graph


if __name__ == '__main__':
    # print(get_dataset('bodyfat.csv'))
    dataset = get_dataset('bodyfat.csv')
    # print_stats(get_dataset('bodyfat.csv'), 1)
    print(regression(dataset, cols=[2,3], betas=[0,0,0]))
    print(regression(dataset, cols=[2,3,4], betas=[0,-1.1,-.2,3]))
    # print('')
    # print(regression(dataset, cols=[2,3,4], betas=[0,-1.1,-.2,3]))
    print(gradient_descent(dataset, cols=[2,3], betas=[0,0,0]))
    iterate_gradient(dataset, cols=[1,8], betas=[400,-400,300], T=10, eta=1e-4)
    ### DO NOT CHANGE THIS SECTION ###
    plot_mse()
