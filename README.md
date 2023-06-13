# Simple policies for Joint Pricing and Inventory Management
This repository contains code that supports the paper titled "Simple policies for Joint Pricing and Inventory Management" [[SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4470538)] by Adam N. Elmachtoub, Harsh Sheth and Yeqing Zhou. All numerics contained in the paper can be found in this [notebook](numerics.ipynb).

Contains implementations of continuous review joint pricing and inventory management systems with:
- Demand models [[demand.py](demand.py)]:
  - Two predefined options: Linear and exponential demand.
  - Custom demand models can be added as a PoissonDemand class instance.
- Instance types [[instance.py](instance.py)]:
  - Two instance types: lost sales and backlogging.
- Lead times [[inventory_state.py](inventory_state.py)]
  - Zero lead time: $(s, S, p)$ policies where $s$ is the re-order point, $S$ is the order-upto level and $p$ is the vector of prices for each inventory state.  
  - Exponential lead times (only lost sales): $(r, Q, p)$ policies where $r$ is the reorder point, $Q$ is the order quantity and $p$ is the vector of prices for each inventory state.
