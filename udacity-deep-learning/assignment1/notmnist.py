# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.

# code changed to Python3

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from urllib.request import urlretrieve
import pickle
import IPython
import matplotlib.pyplot as plt

# Config the matlotlib backend as plotting inline in IPython
# %matplotlib inline

url = 'http://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None


def download_progress_hook(count, blockSize, totalSize):
    """A hook to report the progress of a download. This is mostly intended for users with
    slow internet connections. Reports every 1% change in download progress.
    """
    global last_percent_reported
    percent = int(count * blockSize * 100 / totalSize)

    if last_percent_reported != percent:
        if percent % 5 == 0:
            sys.stdout.write("%s%%" % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write(".")
            sys.stdout.flush()

        last_percent_reported = percent


def maybe_download(filename, expected_bytes, force=False):
    """Download a file if not present, and make sure it's the right size."""
    if force or not os.path.exists(filename):
        print('Attempting to download:', filename)
        filename, _ = urlretrieve(url + filename, filename, reporthook=download_progress_hook)
        print('\nDownload Complete!')
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename


train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)

num_classes = 10
np.random.seed(133)


def maybe_extract(filename, force=False):
    root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
    if os.path.isdir(root) and not force:
        # You may override by setting force=True.
        print('%s already present - Skipping extraction of %s.' % (root, filename))
    else:
        print('Extracting data for %s. This may take a while. Please wait.' % root)
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall()
        tar.close()
    data_folders = [
        os.path.join(root, d) for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d))]
    if len(data_folders) != num_classes:
        raise Exception(
            'Expected %d folders, one per class. Found %d instead.' % (
                num_classes, len(data_folders)))
    print(data_folders)
    return data_folders


train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)

#IPython.display.display_png('notMNIST_large/B/MDEtMDEtMDAudHRm.png')
#IPython.display.display_png('notMNIST_large/J/Nng3b2N0IEFsdGVybmF0ZSBSZWd1bGFyLnR0Zg==.png')

image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.


def load_letter(folder, min_num_images):
    """Load the data for a single letter label."""
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
    print(folder)
    num_images = 0
    for image in image_files:
        image_file = os.path.join(folder, image)
        try:
            image_data = (ndimage.imread(image_file).astype(float) -
                          pixel_depth / 2) / pixel_depth
            if image_data.shape != (image_size, image_size):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            dataset[num_images, :, :] = image_data
            num_images = num_images + 1
        except IOError as e:
            print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

    dataset = dataset[0:num_images, :, :]
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' %
                        (num_images, min_num_images))

    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset


def maybe_pickle(data_folders, min_num_images_per_class, force=False):
    dataset_names = []
    for folder in data_folders:
        set_filename = folder + '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            # You may override by setting force=True.
            print('%s already present - Skipping pickling.' % set_filename)
        else:
            print('Pickling %s.' % set_filename)
            dataset = load_letter(folder, min_num_images_per_class)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)

    return dataset_names


train_datasets = maybe_pickle(train_folders, 45000)
test_datasets = maybe_pickle(test_folders, 1800)


def load_dataset(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


# Display a random matrix with a specified figure number and a grayscale colormap
# largeNameA = train_datasets[0]
# print(largeNameA)
# largeDataA = load_dataset(largeNameA)
# img1 = largeDataA[0, :, :]
# plt.matshow(img1, cmap=plt.cm.gray)
# plt.show()
#
# smallNameJ = test_datasets[9]
# print(smallNameJ)
# smallDataJ = load_dataset(smallNameJ)
# img2 = smallDataJ[0, :, :]
# plt.matshow(img2, cmap=plt.cm.gray)
# plt.show()


# Check whether the data is balanced between classes
# for name in train_datasets:
#     dataset = load_dataset(name)
#     print(name, ' size:', dataset.shape)
#
# for name in test_datasets:
#     dataset = load_dataset(name)
#     print(name, ' size:', dataset.shape)


def make_arrays(nb_rows, img_size):
    if nb_rows:
        dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
        labels = np.ndarray(nb_rows, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels


def merge_datasets(pickle_files, train_size, valid_size=0):
    num_classes = len(pickle_files)
    valid_dataset, valid_labels = make_arrays(valid_size, image_size)
    train_dataset, train_labels = make_arrays(train_size, image_size)
    vsize_per_class = valid_size // num_classes
    tsize_per_class = train_size // num_classes

    start_v, start_t = 0, 0
    end_v, end_t = vsize_per_class, tsize_per_class
    end_l = vsize_per_class + tsize_per_class
    for label, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file, 'rb') as f:
                letter_set = pickle.load(f)
                # let's shuffle the letters to have random validation and training set
                np.random.shuffle(letter_set)
                if valid_dataset is not None:
                    valid_letter = letter_set[:vsize_per_class, :, :]
                    valid_dataset[start_v:end_v, :, :] = valid_letter
                    valid_labels[start_v:end_v] = label
                    start_v += vsize_per_class
                    end_v += vsize_per_class

                train_letter = letter_set[vsize_per_class:end_l, :, :]
                train_dataset[start_t:end_t, :, :] = train_letter
                train_labels[start_t:end_t] = label
                start_t += tsize_per_class
                end_t += tsize_per_class
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise

    return valid_dataset, valid_labels, train_dataset, train_labels



# def show_images(dataset, labels, count):
#     for i in range(0,count):
#         print(labels[i])
#         plt.matshow(dataset[i,:,:], cmap=plt.cm.gray)
#     plt.show()


# show_images(train_dataset, train_labels, 3)
# show_images(test_dataset, test_labels, 3)
# show_images(valid_dataset, valid_labels, 3)


pickle_file = 'notMNIST.pickle'

if not os.path.exists(pickle_file):
    train_size = 200000
    valid_size = 10000
    test_size = 10000

    valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
        train_datasets, train_size, valid_size)
    _, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

    try:
      f = open(pickle_file, 'wb')
      save = {
        'train_dataset': train_dataset,
        'train_labels': train_labels,
        'valid_dataset': valid_dataset,
        'valid_labels': valid_labels,
        'test_dataset': test_dataset,
        'test_labels': test_labels,
        }
      pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
      f.close()
    except Exception as e:
      print('Unable to save data to', pickle_file, ':', e)
      raise

def load_datasets(pickle_file):
    statinfo = os.stat(pickle_file)
    print('Compressed pickle size:', statinfo.st_size)

    f = open(pickle_file, 'rb')
    save = pickle.load(f)
    f.close()
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels


train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = load_datasets(pickle_file)
print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)


def sanitize_dataset(dataset, labels, filter_dataset, similarity_epsilon):
    similarity = cosine_similarity(np.reshape(dataset, (dataset.shape[0],-1)), np.reshape(filter_dataset, (filter_dataset.shape[0],-1)))
    same_filter = np.sum(similarity == 1, axis=1) > 0
    similar_filter = np.sum(similarity > 1-similarity_epsilon, axis=1) > 0
    same_count = np.sum(same_filter)
    similar_count = np.sum(similar_filter)
    filtered_dataset = dataset[same_filter==False]
    filtered_labels = labels[same_filter==False]
    return filtered_dataset, filtered_labels, same_count, similar_count

sanit_pickle_file = 'notMNIST_sanit.pickle'

if not os.path.exists(sanit_pickle_file):
    filtered_valid_dataset, filtered_valid_labels, train_valid_same, train_valid_similar = \
        sanitize_dataset(valid_dataset, valid_labels, train_dataset, 0.001)
    print("training-validation: same=", train_valid_same, "similar=", train_valid_similar)
    filtered_test_dataset, filtered_test_labels, train_test_same, train_test_similar = \
        sanitize_dataset(test_dataset, test_labels, train_dataset, 0.001)
    print("training-testing: same=", train_test_same, "similar=", train_test_similar)
    filtered_test_dataset, filtered_test_labels, valid_test_same, valid_test_similar = \
        sanitize_dataset(filtered_test_dataset, filtered_test_labels, filtered_valid_dataset, 0.001)
    print("validation-testing: same=", valid_test_same, "similar=", valid_test_similar)

    try:
      f = open(sanit_pickle_file, 'wb')
      save = {
        'train_dataset': train_dataset,
        'train_labels': train_labels,
        'valid_dataset': filtered_valid_dataset,
        'valid_labels': filtered_valid_labels,
        'test_dataset': filtered_test_dataset,
        'test_labels': filtered_test_labels,
        }
      pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
      f.close()
    except Exception as e:
      print('Unable to save data to', pickle_file, ':', e)
      raise


train_dataset, train_labels, filtered_valid_dataset, filtered_valid_labels, filtered_test_dataset, filtered_test_labels = load_datasets(sanit_pickle_file)
print('Training (sanitized):', train_dataset.shape, train_labels.shape)
print('Validation (sanitized):', filtered_valid_dataset.shape, filtered_valid_labels.shape)
print('Testing (sanitized):', filtered_test_dataset.shape, filtered_test_labels.shape)


def train_model(dataset, labels, size=None):
    maxSize = dataset.shape[0]
    if size is None:
        size = maxSize
    else:
        if size > maxSize:
            size = maxSize
        indices = np.arange(maxSize)
        np.random.shuffle(indices)
        indices = indices[0:size]
        dataset = dataset[indices]
        labels = labels[indices]
    X = np.reshape(dataset, (size,-1))
    y = labels
    lr = LogisticRegression(n_jobs=4)
    lr.fit(X, y)
    return lr


def model_score(model, dataset, labels):
    X = np.reshape(dataset, (dataset.shape[0],-1))
    y = labels
    return model.score(X, y)


def train(size=None):
    if size is None:
        print("Training with all examples:")
    else:
        print("Training with ", size, " examples:")
    model = train_model(train_dataset, train_labels, size)
    print("  validation score: ", model_score(model, valid_dataset, valid_labels))
    print("  test score: ", model_score(model, test_dataset, test_labels))
    print("  validation score (sanitized): ", model_score(model, filtered_valid_dataset, filtered_valid_labels))
    print("  test score (sanitized): ", model_score(model, filtered_test_dataset, filtered_test_labels))

for size in [50, 100, 1000, 5000]:
    train(size)

# training on all examples:
#train()
