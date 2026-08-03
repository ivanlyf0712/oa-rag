from pyvis.network import Network
from grandcypher import GrandCypher
import networkx as nx

# 假设这是你已构建的 GrandCypher 图对象
gc_graph = GrandCypher(your_networkx_graph)

# 1. 创建一个 pyvis 网络对象
net = Network(notebook=True, height="750px", width="100%")

# 2. 从 GrandCypher 内部的 NetworkX 图加载数据
net.from_nx(gc_graph.G)

# 3. 保存并打开交互式图表
net.show("my_graph.html")