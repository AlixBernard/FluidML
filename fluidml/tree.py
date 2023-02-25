import treelib


class Tree(treelib.Tree):
    def __eq__(self, tree2) -> bool:
        if len(self) != len(tree2):
            return False
        for node in self.all_nodes():
            try:
                node2 = tree2[node.identifier]
            except KeyError:
                return False
            if node != node2:
                return False
        return True


class Node(treelib.Node):
    def __eq__(self, node2) -> bool:
        conditions = [
            self.identifier == node2.identifier,
            self.tag == node2.tag,
            self.data == node2.data,
        ]
        return all(conditions)
