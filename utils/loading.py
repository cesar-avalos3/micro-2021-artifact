import pickle

def save_variables(proc, filename):
    with open(filename, 'wb') as f:
        pickle.dump(proc, f)
        
def open_variables(filename):
    with open(filename, 'rb') as f:
        proc = pickle.load(f)
    return proc