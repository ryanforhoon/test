import rdkit.Chem as Chem
from rdkit import DataStructs
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem import MACCSkeys
import numpy as np
import pandas as pd

# # input block
# # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# input file path                                                                                         |
"""
train_path: input path of train set  :: SMILES file :: smi or txt
test_path: input path of test set   :: SMILES file :: smi or txt
label_path: The file has a classification labels in the same molecular order as the test set
            file format:  csv file , the class column is ['class']
"""
train_path = 'train.smi'
test_path = 'test.smi'
label_path = 'valid_pub.csv'

# input the number of Z (Testing process)                                                                 |
Z_number = [1, 2, 3, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4, 4.1, 4.2,
                 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 5, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8,
                 5.9, 6, 6.1,  6.5, 7, 7.5, 8]
# convert fingerprint                                                                                     |
fingerprint = 'morgan'
# fingerprint = 'maccs'
#                                                                                                         |
# # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||


def match_label(path):
    """
    :param path: file path with ['class'] column
    :return: list of matching labels
    """
    csv_file = pd.read_csv(path)
    get_label = csv_file['class'].values.tolist()
    return get_label


def read_data(path):
    """
    :param path: SMILES file path
    :return: list of SMILES
    """
    with open(path) as F:
        lines = F.readlines()
    record_data = []
    for line in lines:
        line = line.split()
        record_data += line
    return record_data


class Convert:
    """
    convert fingerprint
    """
    def __init__(self):
        super(Convert, self).__init__()

    def calculate_fingerprint(self, all_data, finger_name, morgan_radius=2):
        if finger_name == 'morgan':
            get_fingerprint = self.convert_morgan(all_data, morgan_radius)
        else:
            get_fingerprint = self.convert_maccs(all_data)
        return get_fingerprint

    def convert_morgan(self, all_data, morgan_radius=2):
        """
        convert to morgan
        """
        Get_mol = [Chem.MolFromSmiles(i) for i in all_data]
        Get_mogan = [GetMorganFingerprintAsBitVect(i, morgan_radius) for i in Get_mol]
        """print(len(Get_finger[0]))
        print(len(Get_finger))"""
        return Get_mogan

    def convert_maccs(self, all_data):
        """
        convert to maccs
        """
        Get_mol = [Chem.MolFromSmiles(i) for i in all_data]
        Get_maccs = [MACCSkeys.GenMACCSKeys(i) for i in Get_mol]
        """print(len(Get_finger[0]))
        print(len(Get_finger))"""
        return Get_maccs


def cal_similarity(num_1, num_2, set1, set2):
    """

    :param num_1: set1 index
    :param num_2: set2 index
    :param set1: set1
    :param set2: set2
    :return: Fingerprint similarity
    """
    return DataStructs.FingerprintSimilarity(set1[num_1], set2[num_2])


def cal_mean_std(get_finger):
    """
    :param get_finger: train set
    :return: calculate mean and std
    """
    similarity = []
    # mean
    for set_one in range(len(get_finger)):
        for set_two in range(set_one+1, len(get_finger)):
            similarity.append(cal_similarity(set_one, set_two, get_finger, get_finger))
    average = np.array(similarity).sum()/len(similarity)

    # std
    div = 0
    for num in similarity:
        div += (num - average)**2
    std = (div/(len(similarity)-1))**0.5
    return average, std


def set_similarity(train_finger, test_finger):
    """
    :param train_finger: train set
    :param test_finger: test set
    :return: similarity of train set and test set
    """
    single_similarity = []
    all_similarity = []
    for set_one in range(len(test_finger)):
        for set_two in range(len(train_finger)):
            single_similarity.append(cal_similarity(set_one, set_two, test_finger, train_finger))
        single_similarity.sort(reverse=True)
        all_similarity.append(np.mean(single_similarity[:5]))
        single_similarity = []
    return all_similarity


def cal_DM(z, avg, std):
    """
    calculate DM  DM = avg + Z*std
    :param z: z
    :param avg: avg
    :param std: std
    :return: DM
    """
    return avg + z*std


def get_DM_list(avg, std, z_list):
    """
    calculate the DM of the Z list
    :param avg: avg
    :param std: std
    :param z_list: z list
    :return: DM list(ALL)
    """
    dm = []
    for ls in z_list:
        dm.append(cal_DM(ls, avg, std))
    print("Domain list: ", dm)
    return dm


def get_smiles_in_domain(domain, datasets):
    """
    for test set
    :param domain: domain
    :param datasets: test similarity
    :return: Sequence number and similarity in application domain
    """
    record_index = []
    record_similarity = []
    for idx, n in enumerate(datasets):
        if n > domain:
            record_index.append(idx)
            record_similarity.append(n)
    return record_index, record_similarity


# get domain index and probability
def get_all_domain(dm_list, similarity_list):
    """
    :param dm_list:  dm_list
    :param similarity_list:  similarity of test set and train set
    :return:
    """
    idx_record = []
    sim_record = []
    for domain_number in dm_list:
        get_idx, get_sim = get_smiles_in_domain(domain_number, similarity_list)
        idx_record += [get_idx]
        sim_record += [get_sim]
    return idx_record, sim_record


def save_domain(name, test_set, index, prob, label):
    """
    save test set (in domain)
    """
    # get_smiles
    smiles = []
    for i in index:
        smiles += [test_set[i]]

    # get_label
    labels = []
    for i in index:
        labels.append(label[i])

    # save class
    smiles_pd = pd.DataFrame(smiles, columns=['smiles'])
    similarity_pd = pd.DataFrame(prob, columns=['similarity'])
    class_pd = pd.DataFrame(labels, columns=['class'])
    output_csv = pd.concat((smiles_pd, similarity_pd, class_pd), axis=1)
    path_csv = 'Output_data/' + name + '_domain.csv'
    output_csv.to_csv(path_csv, index=False)

    # save smi
    """
    save test SMILES
    """
    path_of_smi = 'Output_data/' + name + '_domain.smi'
    with open(path_of_smi, 'w') as F:
        for i in smiles:
            F.write(i+'\n')



def main_block(path_train, path_test, path_label, finger_name, z_list):

    # read_data
    train_smiles = read_data(path_train)
    test_smiles = read_data(path_test)
    convert_finger = Convert()
    get_train_finger = convert_finger.calculate_fingerprint(train_smiles, finger_name)
    get_test_finger = convert_finger.calculate_fingerprint(test_smiles, finger_name)

    avg, std = cal_mean_std(get_train_finger)

    print("average: ", avg)
    print("standard: ", std)

    get_test_similarity = set_similarity(get_train_finger, get_test_finger)

    dm_list = get_DM_list(avg, std, z_list)
    test_idx, test_similarity = get_all_domain(dm_list, get_test_similarity)
    print(test_idx[-1])
    # save
    the_label = match_label(path_label)
    for index, name in enumerate(z_list):
        save_domain('Z_{}'.format(name), test_smiles, test_idx[index], test_similarity[index], the_label)

    # save_all_data
    with open('Output_data/record_all.txt', 'w') as F:
        for index, name in enumerate(Z_number):
            F.write('SMILES number(test_set) {}:   << {} >>\n'.format(name, len(test_similarity[index])))
            F.write("DM :  {}\n".format(dm_list[index]))
            F.write("DM = {} + {} x {}\n".format(np.around(avg, decimals=5), name, np.around(std, decimals=5)))
            F.write('\n')


if __name__ == "__main__":
    main_block(train_path, test_path, label_path, fingerprint, Z_number)
