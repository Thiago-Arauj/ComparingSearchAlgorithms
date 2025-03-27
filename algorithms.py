import timeit
import matplotlib.pyplot as plt
from time import sleep
import numpy as np
from functools import partial
import random

class Node:
    def __init__(self, value: int):
        self.value = value
        self.left = None
        self.right = None

class Searchers:
    _last_found = None

    @staticmethod
    def _response(found: bool, index: int = None, target: int = None) -> dict:
        return {
            "code": 200 if found else 404,
            "message": "Target found" if found else "Target not in list",
            "content": {"position": index, "target": target} if found else {}
        }

    @staticmethod
    def _plot_scatter(ax, items: list, current_pos: int, target: int, method_name: str):
        """Scatter plot otimizado para listas grandes"""
        ax.clear()
        
        # Tamanhos dinâmicos baseados no tamanho da lista
        base_size = max(1, 80 - 0.07 * len(items))  # Reduz o tamanho para listas grandes
        highlight_size = max(5, 200 - 0.18 * len(items))
        
        # Plot mais eficiente para listas grandes
        if len(items) > 1000:
            # Para listas muito grandes, mostra apenas uma amostra dos pontos
            step = max(1, len(items) // 500)  # Mostra aproximadamente 500 pontos
            x = range(0, len(items), step)
            y = items[::step]
            colors = ['red' if y_val == target else 'royalblue' for y_val in y]
            ax.scatter(x, y, c=colors, s=base_size, alpha=0.6)
        else:
            # Para listas menores, mostra todos os pontos
            ax.scatter(
                range(len(items)),
                items,
                c=['red' if x == target else 'royalblue' for x in items],
                s=base_size,
                alpha=0.6
            )
        
        # Destacar o ponto atual (sempre mostrado, mesmo em amostras grandes)
        if current_pos < len(items):
            ax.scatter(
                [current_pos],
                [items[current_pos]],
                s=highlight_size,
                facecolors='none',
                edgecolors='gold',
                linewidths=2
            )
        
        ax.set_title(f"{method_name}\nTarget: {target}", fontweight='bold')
        ax.set_ylabel("Valor")
        ax.set_xticks([])
        ax.grid(False)
        
        # Tempos de pausa otimizados:
        if len(items) <= 100:
            plt.pause(0.05)  # 50ms para listas pequenas
        elif len(items) <= 1000:
            plt.pause(0.00001)  # 10ms para listas médias
        else:
            plt.pause(0.0000001)  # 1ms para listas grandes (5000+)

    @staticmethod
    def linear(timer: float, items: list, target: int, plot: bool = True, ax=None) -> dict:
        for i, item in enumerate(items):
            if timer > 0: sleep(timer)
            if plot: Searchers._plot_scatter(ax, items, i, target, "BUSCA LINEAR")
            if item == target:
                return Searchers._response(True, i, target)
        return Searchers._response(False)

    @staticmethod
    def linearSentinel(timer: float, items: list, target: int, plot: bool = True, ax=None) -> dict:
        original = items.copy()
        items.append(target)
        index = 0
        while items[index] != target:
            if timer > 0: sleep(timer)
            if plot: Searchers._plot_scatter(ax, original + [target], index, target, "BUSCA COM SENTINELA")
            index += 1
        items.pop()
        return Searchers._response(index < len(original), index if index < len(original) else None), target

    @staticmethod
    def orderedLinear(timer: float, items: list, target: int, plot: bool = True, ax=None) -> dict:
        for i, item in enumerate(items):
            if timer > 0: sleep(timer)
            if plot: Searchers._plot_scatter(ax, items, i, target, "BUSCA LINEAR ORDENADA")
            if item == target:
                return Searchers._response(True, i, target)
            elif item > target:
                break
        return Searchers._response(False)

    @staticmethod
    def binaryIterative(timer: float, items: list, target: int, plot: bool = True, ax=None) -> dict:
        low, high = 0, len(items) - 1
        while low <= high:
            mid = (low + high) // 2
            if plot: Searchers._plot_scatter(ax, items, mid, target, "BUSCA BINÁRIA ITERATIVA")
            if timer > 0: sleep(timer)
            if items[mid] == target:
                return Searchers._response(True, mid, target)
            elif items[mid] < target:
                low = mid + 1
            else:
                high = mid - 1
        return Searchers._response(False)

    @staticmethod
    def binaryRecursive(timer: float, items: list, target: int, plot: bool = True, ax=None) -> dict:
        def _recursive(low, high):
            if low > high: return Searchers._response(False)
            mid = (low + high) // 2
            if plot: Searchers._plot_scatter(ax, items, mid, target, "BUSCA BINÁRIA RECURSIVA")
            if timer > 0: sleep(timer)
            if items[mid] == target:
                return Searchers._response(True, mid, target)
            elif items[mid] < target:
                return _recursive(mid + 1, high)
            else:
                return _recursive(low, mid - 1)
        return _recursive(0, len(items) - 1)

    @staticmethod
    def adaptativeBinary(timer: float, items: list, target: int, plot: bool = True, ax=None) -> dict:
        low, high = 0, len(items) - 1
        if Searchers._last_found is not None and items[Searchers._last_found] == target:
            return Searchers._response(True, Searchers._last_found, target)
        while low <= high:
            mid = (low + high) // 2
            if plot: Searchers._plot_scatter(ax, items, mid, target, "BUSCA BINÁRIA ADAPTATIVA")
            if timer > 0: sleep(timer)
            if items[mid] == target:
                Searchers._last_found = mid
                return Searchers._response(True, mid, target)
            elif items[mid] < target:
                low = mid + 1
            else:
                high = mid - 1
        return Searchers._response(False)

    @staticmethod
    def binarySearchTree(timer: float, target: int, plot: bool = True, ax=None, root: Node = None) -> dict:
        def _search(node, path=[]):
            if node is None: return Searchers._response(False)
            if timer > 0: sleep(timer)
            path.append(node.value)
            if plot: 
                ax.clear()
                ax.scatter(
                    x=range(len(path)),
                    y=path,
                    c=['red' if x == target else 'royalblue' for x in path],
                    s=[200 if x == node.value else 80 for x in path],
                    edgecolors=['gold' if x == node.value else 'black' for x in path],
                    linewidths=2
                )
                ax.set_title(f"BUSCA EM ÁRVORE BST\nTarget: {target}", fontweight='bold')
                ax.set_xlabel("Passo", fontsize=10)
                ax.grid(True, linestyle='--')
                plt.pause(0.5)
            if node.value == target:
                return Searchers._response(True)
            elif node.value < target:
                return _search(node.right, path)
            else:
                return _search(node.left, path)
        return _search(root, [])

    @staticmethod
    def exponentialBinary(timer: float, items: list, target: int, plot: bool = True, ax=None) -> dict:
        if items[0] == target:
            return Searchers._response(True, 0, target)
        bound = 1
        while bound < len(items) and items[bound] <= target:
            if plot: Searchers._plot_scatter(ax, items, bound, target, "BUSCA EXPONENCIAL + BINÁRIA")
            if timer > 0: sleep(timer)
            bound *= 2
        low, high = bound // 2, min(bound, len(items) - 1)
        return Searchers.binaryIterative(timer, items[low:high+1], target, plot, ax)

    @staticmethod
    def build_bst(items: list) -> Node:
        """Constrói uma BST de forma iterativa para evitar recursion depth"""
        if not items:
            return None
        
        unique_items = sorted(list(set(items)))
        root = Node(unique_items.pop(len(unique_items)//2))
        
        # Implementação iterativa para inserção
        for item in unique_items:
            current = root
            while True:
                if item < current.value:
                    if current.left is None:
                        current.left = Node(item)
                        break
                    else:
                        current = current.left
                else:
                    if current.right is None:
                        current.right = Node(item)
                        break
                    else:
                        current = current.right
        return root

    @staticmethod
    def _insert_node(root: Node, value: int) -> None:
        if value < root.value:
            if root.left is None: root.left = Node(value)
            else: Searchers._insert_node(root.left, value)
        else:
            if root.right is None: root.right = Node(value)
            else: Searchers._insert_node(root.right, value)

def generate_random_list(size: int, target: int) -> list:
    if size <= 0:
        return []
    
    # Gera lista com valores únicos (exceto pelo target)
    lista = random.sample(range(1, max(size, target+2)), size-1)

    if target not in lista:
        lista.append(target)

    random.shuffle(lista)
    
    return lista

def run_and_measure(search_func, name, timer_value, items, target, show_plot=True, ax=None):
    """Função auxiliar para medir tempo e plotar"""
    if name == "7. Busca em BST":
        test_func = partial(search_func, timer_value, target=target, plot=False, ax=ax)
    else:
        test_func = partial(search_func, timer_value, items, target, False, ax)
    
    time_taken = timeit.timeit(test_func, number=n_executions, globals=globals()) / n_executions
    time_micro = time_taken * 1_000_000
    
    if show_plot:
        if name == "7. Busca em BST":
            search_func(timer_value, target=target, plot=True, ax=ax)
        else:
            search_func(timer_value, items, target, True, ax)
        ax.set_title(f"{name}\nTempo médio: {time_micro:.2f} µs (n={n_executions})")
        plt.pause(1)
    
    return name, time_micro

if __name__ == "__main__":
    # Configuração
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Tamanhos das listas (mantendo 10, 100, 1000 conforme versão anterior)
    list_sizes = [10, 100, 1000, 5000]
    target = 57  # Valor que existe em todas as listas
    n_executions = 100
    
    # Gerar datasets
    datasets = {
        size: {
            'random': generate_random_list(size, target),
            'sorted': sorted(generate_random_list(size, target))
        } 
        for size in list_sizes
    }
    
    # Lista de algoritmos
    algorithms = [
        (Searchers.linear, "1. Busca Linear"),
        (Searchers.linearSentinel, "2. Busca com Sentinela"),
        (Searchers.orderedLinear, "3. Busca Linear Ordenada"),
        (Searchers.binaryIterative, "4. Busca Binária Iterativa"),
        (Searchers.binaryRecursive, "5. Busca Binária Recursiva"),
        (Searchers.adaptativeBinary, "6. Busca Binária Adaptativa"),
        (partial(Searchers.binarySearchTree, root=None), "7. Busca em BST"),
        (Searchers.exponentialBinary, "8. Busca Exponencial+Binária")
    ]
    
    # Executar testes
    results = {size: {} for size in list_sizes}
    
    for size in list_sizes:
        print(f"\n=== Testando com lista de tamanho {size} ===")
        random_data = datasets[size]['random']
        sorted_data = datasets[size]['sorted']
        
        # Construir BST
        bst_root = Searchers.build_bst(sorted_data.copy())
        algorithms[6] = (partial(Searchers.binarySearchTree, root=bst_root), "7. Busca em BST")
        
        for alg, name in algorithms:
            use_sorted = any(word in name for word in ["Binária", "BST", "Ordenada"])
            current_data = sorted_data if use_sorted else random_data
            
            alg_name, exec_time = run_and_measure(alg, name, 0, current_data, target, True, ax)
            results[size][alg_name] = exec_time
            print(f"{alg_name}: {exec_time:.2f} µs")
    
    # Gráfico comparativo final com as melhorias visuais
    plt.ioff()
    fig_comparativo, ax_comparativo = plt.subplots(figsize=(14, 8))
    
    markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'D']
    colors = plt.cm.tab10.colors
    
    for i, (alg, name) in enumerate(algorithms):
        times = [results[size][name] for size in list_sizes]
        ax_comparativo.plot(list_sizes, times, 
                          marker=markers[i], 
                          color=colors[i],
                          label=name,
                          linestyle='-',
                          markersize=8,
                          linewidth=2)
    
    ax_comparativo.set_xlabel('Tamanho da Lista (escala logarítmica)', fontsize=12)
    ax_comparativo.set_ylabel('Tempo Médio (µs) - escala log', fontsize=12)
    ax_comparativo.set_title('Desempenho de Algoritmos de Busca em Diferentes Tamanhos de Lista', fontsize=14, pad=20)
    ax_comparativo.set_xscale('log')
    ax_comparativo.set_yscale('log')
    ax_comparativo.grid(True, which="both", ls="--", alpha=0.4)
    ax_comparativo.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Adicionar valores nos pontos
    for size in list_sizes:
        for i, (alg, name) in enumerate(algorithms):
            time_val = results[size][name]
            if time_val < 1:  # Para valores muito pequenos
                ax_comparativo.text(size, time_val*1.5, f"{time_val:.1f}", 
                                  fontsize=8, ha='center', va='bottom', color=colors[i])
            else:
                ax_comparativo.text(size, time_val*1.1, f"{time_val:.0f}", 
                                  fontsize=8, ha='center', va='bottom', color=colors[i])
    
    plt.tight_layout()
    plt.show()