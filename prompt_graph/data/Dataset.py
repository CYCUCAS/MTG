from torch_geometric.data import Dataset, Data


class GraphDataset(Dataset):
    def __init__(self, graphs):
        """
        Initialize the GraphDataset
        :param graphs: A list containing graph objects
        """
        super(GraphDataset, self).__init__()
        self.graphs = graphs

    def len(self):
        """
        Return the size of the dataset
        :return: The size of the dataset
        """
        return len(self.graphs)

    def get(self, idx):
        """
        Obtain the graph with index idx
        :param idx: Index
        :return: Graph object
        """
        graph = self.graphs[idx]
        # Preprocessing or feature extraction of graph data can be carried out here
        # For example, if each graph object has node features and edge features, they can be returned
        # return {'node_features': graph.node_features, 'edge_index': graph.edge_index}
        return graph
