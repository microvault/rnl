from typing import Tuple

import numpy as np


class SumTree:
    # Uma estrutura de dados de árvore binária onde o valor do pai é a soma de seus filhos
    def __init__(self, capacity: int) -> None:
        # Número de folhas (nós finais) que contêm experiências
        self.data_pointer = 0
        # Número de experiências
        self.data_length = 0

        # Inicialize a árvore com todos os nós = 0 e inicialize os dados com todos os valores = 0

        # Número de nós folha (nós finais) que contém experiências
        # Deve set potência de 2.
        self.capacity = int(capacity)
        assert self.is_power_of_2(self.capacity), "A capacidade deve set potência de 2."

        # Here a árvore com todos os valores dos nós = 0
        # Lembre-se que estamos em um nó binário (cada nó tem no máximo 2 filhos), então 2x o tamanho da folha (capacidade) - 1 (nó raiz)
        # Nós pais = capacidade - 1
        # Nós folha = capacidade
        self.tree = np.zeros(2 * capacity - 1)

        """ tree:
            0
           / \
          0   0
         / \\ / \
        0  0 0  0  [Tamanho: capacidade] é nesta linha que as pontuações de prioridade são armazenadas
        """

        # Contém as experiências (portanto, o tamanho dos dados é a capacidade)
        self.data = np.zeros(capacity, dtype=object)

    def __len__(self):
        return self.data_length

    @staticmethod
    def is_power_of_2(n: int) -> bool:
        return ((n & (n - 1)) == 0) and n != 0

    def add(self, data: tuple, priority: float) -> None:
        # Look at what index we want to put the experience
        tree_index = self.data_pointer + self.capacity - 1
        # Update data frame
        self.data[self.data_pointer] = data
        # Update the leaf
        self.update(tree_index, priority)
        # Add 1 to data_pointer
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0
        if self.data_length < self.capacity:
            self.data_length += 1

    def update(self, tree_index: int, priority: float):
        # change = new priority score - former priority score
        priority = max(
            priority.item() if isinstance(priority, np.ndarray) else priority, 1e-5
        )
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        # then propagate the change through tree
        while tree_index != 0:
            """
            Here we want to access the line above
            THE NUMBERS IN THIS TREE ARE THE INDEXES NOT THE PRIORITY VALUES

                0
               / \
              1   2
             / \\ / \
            3  4 5  [6]

            If we are in leaf at index 6, we updated the priority score
            We need then to update index 2 node
            """
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, v: float) -> Tuple[int, np.ndarray, int]:
        # Get the leaf_index, priority value of that leaf and experience associated with that index
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \\   / \
        3   4 5   6    -> storing priority for experiences
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_index = 0

        while True:  # the while loop is faster than the method in the reference code
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            # If we reach bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1

        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0]
