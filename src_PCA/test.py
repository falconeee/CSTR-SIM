
def test1():
    import csv
    
    with open('all.csv') as csvfile:
        
        all_dados = list(csv.reader(csvfile, delimiter="\t"))
        all_dados = np.array(all_dados[1:], dtype=np.str)
        
        X = all_dados[:,0:all_dados.shape[1]-1]
        Y = all_dados[:,all_dados.shape[1]]
    return X, Y


def main():
    print('Executing main() ....')
    
    X, Y = test1()
