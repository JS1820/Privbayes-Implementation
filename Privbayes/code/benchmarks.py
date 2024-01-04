from mbi import Dataset
from ektelo import workload
from itertools import combinations

def adult_benchmark(processed_data, data_domain):
    #csv_path = '/Users/arun/Desktop/Privbayes/private-pgm/data/adult.csv'
    #domain_path = '/Users/arun/Desktop/Privbayes/private-pgm/data/adult-domain.json'
    data = process_dataset(processed_data, data_domain)

    projections = []

    # Generate all possible triplets of attributes
    all_attrs = list(data.domain.attrs)
    for i in range(len(all_attrs)):
        for j in range(i + 1, len(all_attrs)):
            for k in range(j + 1, len(all_attrs)):
                proj = (all_attrs[i], all_attrs[j], all_attrs[k])
                projections.append(proj)

    lookup = {}
    for attr in data.domain:
        n = data.domain.size(attr)
        lookup[attr] = workload.Identity(n)

    workloads = []

    for proj in projections:
        W = workload.Kronecker([lookup[a] for a in proj])
        workloads.append((proj, W))
    print(data, workloads)
    return data, workloads

def process_dataset(csv_path, domain_path):

    data = Dataset.load(csv_path, domain_path)

    return data

#if __name__ == '__main__':
#    data, workloads = adult_benchmark()
#    print(workloads)
