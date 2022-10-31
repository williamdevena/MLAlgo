import numpy as np

def perceptron2(epochs, points, labels, lr):
    w=np.zeros(points[0].shape[0])
    #print(w)
    
    for epoch in range(epochs):
        for index in range(len(points)):
            point = points[index]
            label = labels[index]
            out = np.dot(point, w)
            if out>=0:
                out = 1
            else:
                out = -1
            if out != label:
                w = w + (label * point)

    return w


def main():      
    c1 = np.array([1,1,1])
    c2 = np.array([1,2,2])
    c3 = np.array([1,3,3])
    t1 = np.array([1,3,4])
    t2 = np.array([1,3,6])
    t3 = np.array([1,4,6])


    labels = [-1, -1, -1, 1, 1, 1]

    points = np.array([c1,c2,c3,t1,t2,t3])

    epochs = 100
    lr = 0.2

    print(perceptron2(epochs, points, labels, lr))



if __name__ == "__main__":
    main()