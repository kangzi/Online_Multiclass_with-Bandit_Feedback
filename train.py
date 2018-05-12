from sklearn.feature_extraction.text import TfidfVectorizer
import scipy
import numpy as np
import data
import pickle
import argparse
import banditron
import confidit
import bpa
from statistics import mean, stdev
import confidit_kernel
import banditron_kernel
import bpa_kernel

class conditional_error(Exception):
    pass


parser = argparse.ArgumentParser(description='Online Complementary Learning.')
parser.add_argument('--dataset', '-d', type=str, default='mnist',
                    help='Choose dataset (default: mnist)')
parser.add_argument('--algorithm', '-a', type=str, default='banditron',
                    help='Choose an algorithm from \n'+
                         'banditron \n' +
                         'confidit \n' +
                         '(default: banditron)')
parser.add_argument('--kernel', '--k', action='store_true',
                    help='Determine whether kernel is used.')
parser.add_argument('--bag_size', '-b', type=int, default=500,
                    help='Determine the size of bags.')
parser.add_argument('--G', '-g', type=float, default=1.0,
                    help='hyper parameter for kernel')
parser.add_argument('--complementary', '--c', action='store_true',
                    help='Use complementary or not')
parser.add_argument('--mistake_pass', '--m', action='store_true',
                    help='Use mistake_pass or not')
parser.add_argument('--adaptive_train', '--a', action='store_true',
                    help='Use adaptive train or not')
parser.add_argument('--seed1', '-s1', type=int, default=0,
                    help='seed for start')
parser.add_argument('--seed2', '-s2', type=int, default=1,
                    help='seed for end')
parser.add_argument('--f_name', '-f', type=str, default='register',
                    help='the name of result file')


def make_reuters_data(x, y):
    yy = np.zeros(x.shape[0])
    for cl in classes:
        lb_num = np.argmax(labels == cl)
        yy += (y[:, lb_num] == 1)

    yy = (yy == 1)
    x_data = []
    y_data = []
    index = 0
    for cl in classes:
        lb_num = np.argmax(labels == cl)
        x_cl = x[np.logical_and(yy, (y[:, lb_num] == 1))]
        x_data.append(x_cl)
        y_data.append(np.ones(x_cl.shape[0]) * index)
        index += 1

    x = x_data[0]
    for i in range(1, len(x_data)):
        x = np.r_[x, x_data[i]]

    y = y_data[0]
    for i in range(1, len(y_data)):
        y = np.r_[y, y_data[i]]

    seq = np.arange(x.shape[0])
    np.random.shuffle(seq)
    x = x[seq]
    y = y[seq].astype(np.int64)

    return x, y


if __name__ == '__main__':
    args = parser.parse_args()
    seed1 = args.seed1
    seed2 = args.seed2

    if args.dataset == '20news':
        K = 20
        x_train, y_train = data.get_data('./data/news20.scale')

    elif args.dataset == 'reuters':
        K = 20
        from nlp import reuters
        datasets = reuters.load_data()
        labels = np.array(datasets['labels'])
        classes = ['acq', 'alum', 'cocoa', 'coffee', 'copper', 'cpi', 'crude', 'earn', 'gnp', 'gold', 'grain',
                   'interest', 'jobs', 'money-fx',
                   'money-supply', 'reserves', 'rubber', 'ship', 'sugar', 'trade']

        x_train = np.array(datasets['x_train'])
        x_test = np.array(datasets['x_test'])
        y_train = np.array(datasets['y_train'])
        y_test = np.array(datasets['y_test'])

        # make train data
        x_train, y_train = make_reuters_data(x_train, y_train)
        # make test data
        x_test, y_test = make_reuters_data(x_test, y_test)

    elif args.dataset == 'usps':
        K = 10
        x_train, y_train = data.get_data('./data/usps')
        x_test, y_test = data.get_data('./data/usps.t')
        x_train /= 255.0
        x_test /= 255.0

    elif args.dataset == 'letter':
        K = 26
        x_train, y_train = data.get_data('./data/letter.scale')
        x_test, y_test = data.get_data('./data/letter.scale.t')

    elif args.dataset == 'satimage':
        K = 6
        x_train, y_train = data.get_data('./data/satimage.scale')
        x_test, y_test = data.get_data('./data/satimage.scale.t')
        print(np.max(y_train), np.min(y_train))

    elif args.dataset == 'vowel':
        K = 11
        x_train, y_train = data.get_data('./data/vowel.scale')
        x_test, y_test = data.get_data('./data/vowel.scale.t')

    elif args.dataset == 'shuttle':
        K = 7
        x_train, y_train = data.get_data('./data/shuttle.scale')
        x_test, y_test = data.get_data('./data/shuttle.scale.t')

    elif args.dataset == 'sector':
        K = 105
        x_train, y_train = data.get_data('./data/sector.scale')
    elif args.dataset == 'pendigits':
        K = 10
        x_train, y_train = data.get_data('./data/pendigits')
        x_test, y_test = data.get_data('./data/pendigits.t')
        x_train /= 100.0
        x_test /= 100.0
    elif args.dataset == 'Sensorless':
        K = 11
        x_train, y_train = data.get_data('./data/Sensorless.scale')
    elif args.dataset == 'mnist':
        K = 10
        save_file = './data/mnist.pkl'
        with open(save_file, 'rb') as f:
            dataset = pickle.load(f)
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0
        x_train, y_train = dataset['train_img'], dataset['train_label']
        x_test, y_test = dataset['test_img'], dataset['test_label']
    else:
        raise conditional_error

    if np.min(y_train) == 1:
        y_train -= 1

    if args.kernel:
        if args.algorithm == 'banditron' or 'bpa':
            parameter_list = [0.001, 0.025, 0.1, 0.3, 0.5]
        else:
            parameter_list = [0.0001, 0.01, 1.0, 100.0, 10000.0]
    else:
        if args.algorithm == 'banditron' or 'bpa':
            parameter_list = [0.001, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        else:
            parameter_list = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0]

    N = x_train.shape[0]

    if args.kernel:
        g_list = [0.01, 0.1, 1.0, 10.0, 100.0]
        final_p_list = []
        for par in parameter_list:
            for g in g_list:
                np.random.seed(0)
                if args.algorithm == 'banditron':
                    Band = banditron_kernel.Banditron_kernel(x_train, y_train, g = g, B = args.bag_size,
                                                             gamma=par, test_interval=100)
                    l, acc, final_p, _ = Band.train(N)
                elif args.algorithm == 'confidit':
                    Conf = confidit_kernel.Confidit_kernel(x_train, y_train, g = g, B = args.bag_size, eta=par, test_interval=100)
                    l, acc, final_p, _ = Conf.train(N)
                elif args.algorithm == 'bpa':
                    BPA = bpa_kernel.BPA_kernel(x_train, y_train, g = g, B = args.bag_size, gamma=par, test_interval=100)
                    l, acc, final_p, _ = BPA.train(N)
                else:
                    raise conditional_error
                final_p_list.append(final_p)
            best_parameter = parameter_list[max(zip(final_p_list, range(len(final_p_list))))[1] // 5]
            best_g = g_list[max(zip(final_p_list, range(len(final_p_list))))[1] % 5]
    else:
        final_p_list = []
        for par in parameter_list:
            np.random.seed(0)
            if args.algorithm == 'banditron':
                Band = banditron.Banditron(x_train, y_train, gamma=par, test_interval=100)
                l, acc, final_p, final_ac = Band.train(N)
            elif args.algorithm == 'confidit':
                Conf = confidit.Confidit(x_train, y_train, eta=par, test_interval=100)
                l, acc, final_p, final_ac = Conf.train(N)
            elif args.algorithm == 'bpa':
                BPA = bpa.BPA(x_train, y_train, gamma=par, test_interval=100)
                l, acc, final_p, final_ac = BPA.train(N)
            else:
                raise conditional_error
            final_p_list.append(final_p)
        best_parameter = parameter_list[max(zip(final_p_list, range(len(final_p_list))))[1]]

    acc = []
    cum_l, cum_ac = [], []
    l_list, ac_list = [], []
    for p in range(seed1, seed2):
        np.random.seed(p)
        if args.kernel:
            if args.algorithm == 'banditron':
                Band = banditron_kernel.Banditron_kernel(x_train, y_train, g = best_g, gamma=best_parameter, test_interval=100)
                l, acc, final_p, final_ac = Band.train(N)
            elif args.algorithm == 'confidit':
                Conf = confidit_kernel.Confidit_kernel(x_train, y_train, g = best_g, eta=best_parameter, test_interval=100)
                l, acc, final_p, final_ac = Conf.train(N)
            elif args.algorithm == 'bpa':
                BPA = bpa_kernel.BPA_kernel(x_train, y_train, g = best_g, gamma=best_parameter, test_interval=100)
                l, acc, final_p, final_ac = BPA.train(N)
            else:
                raise conditional_error
        else:
            if args.algorithm == 'banditron':
                Band = banditron.Banditron(x_train, y_train, gamma=best_parameter, test_interval=100)
                l, acc, final_p, final_ac = Band.train(N)
            elif args.algorithm == 'confidit':
                Conf = confidit.Confidit(x_train, y_train, eta=best_parameter, test_interval=100)
                l, acc, final_p, final_ac = Conf.train(N)
            elif args.algorithm == 'bpa':
                BPA = bpa.BPA(x_train, y_train, gamma=best_parameter, test_interval=100)
                l, acc, final_p, final_ac = BPA.train(N)
            else:
                raise conditional_error

        l_list.append(final_p)
        ac_list.append(final_ac)

        if cum_l == []:
            cum_l = l
            cum_ac = acc
        else:
            cum_l = [x + y for (x, y) in zip(cum_l, l)]
            cum_ac = [x + y for (x, y) in zip(cum_ac, acc)]

    cum_l = [x / (seed2 - seed1) for x in cum_l]
    cum_ac = [x / (seed2 - seed1) for x in cum_ac]

    if args.kernel:
        with open('./result/'+args.f_name+'_'+args.algorithm+'_kernel_l.pickle', 'wb') as write_f:
            pickle.dump(cum_l, write_f)
        with open('./result/' + args.f_name + '_' + args.algorithm + '_kernel_ac.pickle', 'wb') as write_f:
            pickle.dump(cum_ac, write_f)
    else:
        with open('./result/'+args.f_name+'_'+args.algorithm+'_l.pickle', 'wb') as write_f:
            pickle.dump(cum_l, write_f)
        with open('./result/' + args.f_name + '_' + args.algorithm + '_ac.pickle', 'wb') as write_f:
            pickle.dump(cum_ac, write_f)

    print("ordinary label mean", mean(l_list))
    print("ordinary label stdev", stdev(l_list))
    print("predict mean", mean(ac_list))
    print("predict stdev", stdev(ac_list))