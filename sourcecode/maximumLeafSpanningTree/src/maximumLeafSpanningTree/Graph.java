package maximumLeafSpanningTree;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.PriorityBlockingQueue;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.function.Consumer;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class Graph {
	private final LinkedList<Integer>[] adj;
	private final int n; // number of vertices
	private final Object lock = new Object();

	@SuppressWarnings("unchecked")
	public Graph(int n) {
		if (n < 0) {
			throw new IllegalArgumentException();
		}

		this.n = n;
		this.adj = (LinkedList<Integer>[]) new LinkedList[n];
		for (int i = 0; i < n; i++) {
			adj[i] = new LinkedList<Integer>();
		}
	}

	/**
	 * Create a graph with n vertices and probability of edge p
	 * @param n 
	 * @param p
	 */
	@SuppressWarnings("unchecked")
	public Graph(int n, double p) {
		this.adj = (LinkedList<Integer>[]) new LinkedList[n];
		this.n = n;
		for (int i = 0; i < n; i++) {
			adj[i] = new LinkedList<Integer>();
		}
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				if (i == j)
					continue;
				if (p + Math.random() >= 1) {
					addEdge(i, j);
				}
			}
		}
	}
	
	public static Graph copy(Graph g) {
		var newG = new Graph(g.n);
		for (int i = 0; i < g.n; i++) {
			var edges = g.edgeIterator(i);
			while(edges.hasNext()) {
				newG.addEdge(i, edges.next());
			}
		}
		return newG;
	}

	public static void main(String[] args) throws InterruptedException {

        if (args.length != 2 && args.length != 1) {
            System.err.println("Usage: java MainApp <integer> <\"2-app\"|\"3-app\">");
            return;
        }

        int size;
        try {
            size = Integer.parseInt(args[0]);
            if (size <= 0) {
                throw new IllegalArgumentException("The first argument must be an integer greater than 0.");
            }
        } catch (Exception e) {
            System.err.println("Error: The first argument must be an integer greater than 0.");
            return;
        }

		System.out.println("... Creating the graph ... probability of adding an edge is 0.14");

        var g = new Graph(size, 0.14);
		var L_opt = g.numberOfVertices()-1; 
		Graph g_prime = null;
		long startTime;
		long endTime;
		long timeElapsed = -1;
		
        if(args.length==2) {
            String approximationType = args[1];
            if (!approximationType.equals("2-app") && !approximationType.equals("3-app")) {
                System.err.println("Error: The second argument must be either \"2-app\" or \"3-app\".");
                System.exit(1);
                return;
            }

            switch (approximationType) {
                case "2-app":
                    System.out.println("Running 2-approximation algorithm...");
            		startTime = System.nanoTime();
                    g_prime = Graph.approximationSolisOba(g);
            		endTime = System.nanoTime();
            		timeElapsed = endTime - startTime;
                    break;
                case "3-app":
                    System.out.println("Running 3-approximation algorithm...");
            		startTime = System.nanoTime();
                    g_prime = Graph.approximationLuAndRavi(g);
            		endTime = System.nanoTime();
            		timeElapsed = endTime - startTime;
                    break;
            }

    		System.out.println("L_opt="+L_opt);
    		System.out.println("L_app="+g_prime.getLeafs().size());
    		System.out.println("Ratio for "+approximationType+" : "+g.n/g_prime.getLeafs().size());
    		System.out.println("Execution time : "+timeElapsed/Math.pow(10, 6)+"ms");

        }else {
        	var g2 = Graph.copy(g);
            System.out.println("Running 2-approximation algorithm...");
    		startTime = System.nanoTime();
            g_prime = Graph.approximationSolisOba(g);
    		endTime = System.nanoTime();
    		timeElapsed = endTime - startTime;

    		System.out.println("L_opt="+L_opt);
    		System.out.println("L_2-app="+g_prime.getLeafs().size());
    		System.out.println("Ratio for 2-app : "+g.n/g_prime.getLeafs().size());
    		System.out.println("Execution time : "+timeElapsed/Math.pow(10, 6)+"ms");
            
            System.out.println("Running 3-approximation algorithm...");
    		startTime = System.nanoTime();
            g_prime = Graph.approximationLuAndRavi(g2);
    		endTime = System.nanoTime();
    		timeElapsed = endTime - startTime;

    		System.out.println("L_opt="+L_opt);
    		System.out.println("L_3-app="+g_prime.getLeafs().size());
    		System.out.println("Ratio for 3-app : "+g.n/g_prime.getLeafs().size());
    		System.out.println("Execution time : "+timeElapsed/Math.pow(10, 6)+"ms");
        }
	}

	/**
	 * The number of vertices
	 * @return n
	 */
	public int numberOfVertices() {
		return n;
	}

	/**
	 * Add an edge
	 * @param i
	 * @param j
	 */
	public void addEdge(int i, int j) {
		Objects.checkIndex(i, n);
		Objects.checkIndex(j, n);
		synchronized (lock) {
			if (!adj[i].contains(j)) {
				adj[i].add(j);
				adj[j].add(i);
			}
		}
	}

	/**
	 * Clear the edges of i for u=i and u->v
	 * @param i
	 */
	public void simpleClearEdges(int i) {
		Objects.checkIndex(i, n);
		synchronized (lock) {
			adj[i].clear();
		}
	}

	/**
	 * Removes u->v for u=i and v=j
	 * @param i
	 * @param j
	 */
	public void removeSingleEdge(int i, int j) {
		Objects.checkIndex(i, n);
		Objects.checkIndex(j, n);
		synchronized (lock) {
			adj[i].remove((Object) j);
		}
	}

	/**
	 * Get degree of i
	 * @param i
	 * @return
	 */
	public int getDegree(int i) {
		Objects.checkIndex(i, n);
		synchronized (lock) {
			return adj[i].size();
		}
	}

	/**
	 * Get an iterator of existing edges incident to i
	 * @param i
	 * @return
	 */
	public Iterator<Integer> edgeIterator(int i) {
		Objects.checkIndex(i, n);
		synchronized (lock) {
			return new Iterator<Integer>() {
				private int cursor = 0;

				@Override
				public Integer next() {
					if (!hasNext()) {
						throw new NoSuchElementException();
					}
					var edge = adj[i].get(cursor);
					cursor++;
					return edge;
				}

				@Override
				public boolean hasNext() {
					return cursor < adj[i].size();
				}
			};
		}
	}

	/**
	 * Execute a consumer on each incident edges of i
	 * @param i
	 * @param consumer
	 */
	public void forEachEdge(int i, Consumer<Integer> consumer) {
		Objects.checkIndex(i, n);
		Objects.requireNonNull(consumer);
		synchronized (lock) {
			edgeIterator(i).forEachRemaining(consumer);
		}
	}

	/**
	 * The graph in string
	 * @return
	 */
	public String toVisualGraph() {
		synchronized (lock) {
			var sb = new StringBuilder();
			sb.append("g [|V|=");
			sb.append(n);
			sb.append("] {\n");
			for (int i = 0; i < adj.length; i++) {
				var edges = adj[i];

				sb.append(i).append(";\n");
				for (var j : edges) {
					sb.append(i).append(" -> ").append(j).append(" ;\n");
				}
			}
			sb.append("}");

			return sb.toString();
		}
	}

	/**
	 * Enum for 2-approximation algorithm
	 */
	enum Rule {
		None(20),
		Rule1(1), // x->y->[ab]
		Rule2(2), // x->y->[abcd]
		Rule3(3), // x->[abcd]
		Rule4(4); // x->y

		private final int priority;

		Rule(int priority) {
			this.priority = priority;
		}

		public int getPriority() {
			return priority;
		}
	}

	/**
	 * Return the rule to apply for x according to g and the tree
	 * @param tree
	 * @param x
	 * @return
	 */
	public Rule whichRule(Graph tree, int x) {
		synchronized (lock) {
			var neighborsOutOfTree = getNeighborsOutOfTree(tree, x);
			if (neighborsOutOfTree.length == 1) {
				var y = neighborsOutOfTree[0];
				var childsOfY = getNeighborsOutOfTree(tree, y);
				return switch (childsOfY.length) {
				case 0: { // x->y
					yield Rule.Rule4;
				}
				case 1: {
					yield Rule.None;
				}
				case 2: {
					yield Rule.Rule1;
				}
				default:
					yield Rule.Rule2;
				};
			} else if (neighborsOutOfTree.length > 1) {
				return Rule.Rule3;
			}
			return Rule.None;
		}
	}

	/**
	 * Return the neighbors out of the tree for x according to g and the tree
	 * @param tree
	 * @param x
	 * @return
	 */
	public int[] getNeighborsOutOfTree(Graph tree, int x) {
		synchronized (lock) {
			int[] neighborsOutOfTree = new int[getDegree(x)];
			int neighborIndex = 0;
			var neighbors = edgeIterator(x);
			while (neighbors.hasNext()) {
				var neighbor = neighbors.next();
				if (tree.getDegree(neighbor) == 0) {
					neighborsOutOfTree[neighborIndex] = neighbor;
					neighborIndex++;
				}
			}
			return Arrays.copyOf(neighborsOutOfTree, neighborIndex);
		}
	}

	/**
	 * Return the neighbors out of the tree for x according to g and the tree and the rule
	 * @param tree
	 * @param x
	 * @return
	 */
	public int[] getNeighborsByRule(Rule rule, Graph tree, int x) {
		synchronized (lock) {
			return switch (rule) {
			case Rule1, Rule2 -> {
				var y = getNeighborsOutOfTree(tree, x); // by definition, x has only one neighboor out of tree
				var childsOfY = getNeighborsOutOfTree(tree, y[0]);
				yield Stream.concat(Stream.of(y), Stream.of(childsOfY)).flatMapToInt(Arrays::stream).toArray();
			}
			case Rule3, Rule4 -> {
				yield getNeighborsOutOfTree(tree, x);
			}
			default -> {
				yield new int[] {};
			}
			};
		}
	}

	/**
	 * Return the leaves
	 * @return
	 */
	public List<Integer> getLeafs() {
		var leafs = new ArrayList<Integer>();
		for (int i = 0; i < n; i++) {
			if (getDegree(i) == 1) {
				leafs.add(i);
			}
		}
		return leafs;
	}

	/**
	 * The 2-approximation algorithm
	 * @param g
	 * @return
	 * @throws InterruptedException
	 */
	static public Graph approximationSolisOba(Graph g) throws InterruptedException {
		ArrayList<Graph> forest = new ArrayList<>();
		ArrayList<Graph> incidentEdgesForest = new ArrayList<>();
		PriorityBlockingQueue<Graph.LeafRule> expandLeaves = new PriorityBlockingQueue<>(g.n / 2, LeafRule::compare);
		ArrayBlockingQueue<Integer> checkLeaves = new ArrayBlockingQueue<Integer>(g.n / 2);

		for (int root = 0; root < g.n; root++) { // N
			if (g.getDegree(root) >= 3) {

				Graph tree = new Graph(g.n);
				Graph incident_edges = new Graph(g.n);
				checkLeaves.put(root);

				// Thread--
				Thread expandthread = Thread.ofPlatform().start(() -> {
					while (!Thread.interrupted()) {
						// "Consume" expanding leaves
						try {
							var x = expandLeaves.poll(); // N
							if(x==null) {
								continue;
							}
							// Apply rule
							var newNeighbors = g.getNeighborsByRule(x.rule, tree, x.leaf); // M

							switch (x.rule) {
							case Rule1, Rule2:
								tree.addEdge(x.leaf, newNeighbors[0]);
								for (int i = 1; i < newNeighbors.length; i++) {
									tree.addEdge(newNeighbors[0], newNeighbors[i]);
									// "Produce" check leaves
									checkLeaves.put(newNeighbors[i]);
								}
								break;
							case Rule3, Rule4:
								for (var vertice : newNeighbors) {
									tree.addEdge(x.leaf, vertice);
									// "Produce" check leaves
									checkLeaves.put(vertice);
								}
								break;
							default:
								break;
							}

						} catch (InterruptedException e) {
							Thread.currentThread().interrupt();
							break;
						}

						// Update existing leaves
						LeafRule[] leavesCopy = expandLeaves.toArray(new LeafRule[0]); // Deep copy of the LeafRules

						expandLeaves.clear();
						for (var leaf : leavesCopy) { // N
							var rule = g.whichRule(tree, leaf.leaf);
							if (rule != Rule.None) {
								leaf.rule = rule;
								expandLeaves.add(leaf);
							}
						}
					}
					return;
				});
				// --Thread

				while (!expandthread.isInterrupted()) {
					synchronized (expandLeaves) {
					    synchronized (checkLeaves) {
					        if (expandLeaves.isEmpty() && checkLeaves.isEmpty()) {
					            expandthread.interrupt();
					            break;
					        }
					    }
					}
					var x = checkLeaves.take(); // N
					var rule = g.whichRule(tree, x);
					if (rule != Rule.None) {
						// "Produce" expanding leaves
						expandLeaves.add(new LeafRule(x, rule));
					}
				}
				
				expandthread.interrupt();
				forest.add(tree);

				for (int i = 0; i < tree.n; i++) { // N
					if (tree.getDegree(i) > 0) {
						g.simpleClearEdges(i);
					}
				}

				for (int u = 0; u < g.n; u++) { // N
					if (g.getDegree(u) > 0) {
						var vertices = g.adj[u].stream().toList();
						for (var v : vertices) { // M
							if (tree.getDegree(v) > 0) {
								incident_edges.addEdge(u, v);
								g.removeSingleEdge(u, v);
							}
						}
					}
				}
				incidentEdgesForest.add(incident_edges);
			}
		}

		// CREATE MLST
		var mergeThreads = new Thread[forest.size()];
		var mlst = new Graph(g.n);

		IntStream.range(0, forest.size()).forEachOrdered((nthTree) -> {
			mergeThreads[nthTree] = Thread.ofPlatform().start(() -> {
				var tree = forest.get(nthTree);
				for (int u = 0; u < tree.n; u++) { // N
					var uIterator = tree.edgeIterator(u);
					while (uIterator.hasNext()) { // M
						mlst.addEdge(u, uIterator.next());
					}
				}
			});
		});

		var computed = computeLinkingEdges(forest, incidentEdgesForest);

		for (Thread thread : mergeThreads) {
			thread.join();
		}

		mlst.addLinkingEdges(computed);

		return mlst;
	}

	/**
	 * Encapsulation of a leaf and its rule
	 */
	static class LeafRule {
		int leaf;
		Rule rule;

		public LeafRule(int leaf, Rule rule) {
			this.leaf = leaf;
			this.rule = rule;
		}

		public static int compare(LeafRule e1, LeafRule e2) {
			return Integer.compare(e1.rule.getPriority(), e2.rule.getPriority());
		}

		@Override
		public String toString() {
			return "[Leaf: " + leaf + ", rule: " + rule + "]";
		}
	}

	/**
	 * DFS
	 * @param g
	 * @param u
	 * @return
	 */
	static public Graph DFS(Graph g, int u) {
		var visitedGraph = new Graph(g.n);
		var visited = new BitSet(g.n);
		visited.flip(u);
		g.forEachEdge(u, (v) -> {
			if (!visited.get(v)) {
				visited.set(v);
				visitedGraph.addEdge(u, v);
				DFS_rec(g, v, visited, visitedGraph);
			}
		});
		return visitedGraph;
	}

	/**
	 * DFS that returns the components
	 * @param g
	 * @return
	 */
	static public List<Graph> DFS_trees(Graph g) {
		var list = new ArrayList<Graph>();
	    var visited = new BitSet(g.n);

	    for (int u = 0; u < g.n; u++) {
	        if (!visited.get(u)) {
	            visited.set(u);
	            if(g.getDegree(u)==0)continue;
	            var currentTree = new Graph(g.n);
	            DFS_rec(g, u, visited, currentTree);
	            list.add(currentTree);
	        }
	    }
	    return list;
	}


	static private void DFS_rec(Graph g, int u, BitSet visited, Graph visitedGraph) {
		g.forEachEdge(u, (v) -> {
			if (!visited.get(v)) {
				visited.set(v);
				visitedGraph.addEdge(u, v);
				DFS_rec(g, v, visited, visitedGraph);
			}
		});
	}

	/**
	 * Union between two trees for a balanced tree
	 * @param x
	 * @param y
	 * @param parents
	 * @param heights
	 * @param subset
	 */
	static public void union(int x, int y, int[] parents, int[] heights, ConcurrentHashSet[] subset) {
		int px = find(x, parents);
		int py = find(y, parents);
		if (heights[px] <= heights[py]) {
			parents[px] = py;
			if (heights[px] == heights[py]) {
				heights[py]++;
			}
			subset[py].addAll(subset[px].get());
		} else {
			parents[py] = px;
			subset[px].addAll(subset[py].get());
		}
	}

	/**
	 * Find the parent of x, and update the parent
	 * @param x
	 * @param parents
	 * @return
	 */
	static public int find(int x, int[] parents) {
		Objects.checkIndex(x, parents.length);
		if (parents[x] == x) {
			return x;
		}
		parents[x] = find(parents[x], parents);
		return parents[x];
	}

	/**
	 * The 3-approximation algorithm
	 * @param g
	 * @return
	 * @throws InterruptedException
	 */
	static public Graph approximationLuAndRavi(Graph g) throws InterruptedException {
		var forest = new Graph(g.n);
		var parents = new int[g.n];
		var heights = new int[g.n];
		var d = new AtomicIntegerArray(g.n);
		var subsets = new ConcurrentHashSet[g.n];
		var subsetLocks = new Object[g.n];

		var verticeThreads = new Thread[g.n / 2];
		var verticeCounter = new AtomicInteger();

		// initialize
		for (int i = 0; i < g.n; i++) {
			subsets[i] = new ConcurrentHashSet();
			subsets[i].add(i);
			parents[i] = i;
			subsetLocks[i] = new Object();
		}

		IntStream.range(0, verticeThreads.length).forEachOrdered(i -> {
			verticeThreads[i] = Thread.ofPlatform().start(() -> {
				while (true) {
					int v = verticeCounter.getAndIncrement();
					if (v >= g.n) return;

					var set_prime = new HashSet<Integer>();
					var d_prime = 0;
					var neighbors = g.edgeIterator(v);
					while (neighbors.hasNext()) {
						var u = neighbors.next();
						int min = Math.min(v, u);
						int max = Math.max(v, u);
						synchronized (subsetLocks[min]) {
							synchronized (subsetLocks[max]) {
								var pu = find(u, parents);
								boolean uInSetPrime = set_prime.stream().map(x -> find(x, parents) == pu).anyMatch(r -> r == true);
								if (find(u, parents) != find(v, parents) && !uInSetPrime) {
									d_prime++;
									set_prime.addAll(subsets[find(u, parents)].get());
								}
							}
						}
					}

					if (d.get(v) + d_prime >= 3) {
						var set_primeIterator = set_prime.iterator();
						while (set_primeIterator.hasNext()) {
							var u = set_primeIterator.next();
							int min = Math.min(v, u);
							int max = Math.max(v, u);

							synchronized (subsetLocks[min]) {
								synchronized (subsetLocks[max]) {
									if (find(u, parents) != find(v, parents)) {
										forest.addEdge(u, v);
										union(u, v, parents, heights, subsets);
										d.incrementAndGet(v);
										d.incrementAndGet(u);
									}
								}
							}
						}
					}
				}
			});
		});

		for (int i = 0; i < verticeThreads.length; i++) {
			verticeThreads[i].join();
		}


		// From forest graph to list of trees
		var trees = DFS_trees(forest);

		// Compute forest of incident edges for each tree
		var incidentEdgesForest = new ArrayList<Graph>(trees.size());
		for (int i = 0; i < trees.size(); i++) {
			incidentEdgesForest.add(new Graph(g.n));
		}

		for (int i = 0; i < trees.size(); i++) {
			var currentTree = trees.get(i);
			var incidentEdgesCurrentTree = incidentEdgesForest.get(i);
			var leafs = currentTree.getLeafs();
			for (var leaf : leafs) {
				if (g.getDegree(leaf) > 1) { // has outward edges out of tree
					var edges = g.getNeighborsOutOfTree(currentTree, leaf);
					for (var v : edges) {
						incidentEdgesCurrentTree.addEdge(leaf, v);
					}
				}
			}
		}


		// CREATE MLST
		var mergeThreads = new Thread[trees.size()];
		var mlst = new Graph(g.n);

		IntStream.range(0, trees.size()).forEachOrdered((nthTree) -> {
			mergeThreads[nthTree] = Thread.ofPlatform().start(() -> {
				var tree = trees.get(nthTree);
				for (int u = 0; u < tree.n; u++) { // N
					var uIterator = tree.edgeIterator(u);
					while (uIterator.hasNext()) { // M
						mlst.addEdge(u, uIterator.next());
					}
				}
			});
		});

		var computed = computeLinkingEdges(trees, incidentEdgesForest);

		for (Thread thread : mergeThreads) {
			thread.join();
		}

		mlst.addLinkingEdges(computed);
		return mlst;
	}

	/**
	 * Find which tree is connected to which and which edge to add for that
	 * @param forest
	 * @param incidentEdgesForest
	 * @return
	 */
	static private SpanningPath computeLinkingEdges(List<Graph> forest, List<Graph> incidentEdgesForest) {
		var incidentEdgesGraph = new Graph(forest.size());
		@SuppressWarnings("unchecked")
		var incidentEdgesValues = (HashMap<Integer, Integer[]>[]) new HashMap[forest.size()];
		for (int i = 0; i < forest.size(); i++) {
			incidentEdgesValues[i] = new HashMap<Integer, Integer[]>();
		}
		var treatedTrees = new BitSet[forest.size()];
		for (int i = 0; i < forest.size(); i++) {
			treatedTrees[i] = new BitSet(forest.size());
		}
		
		outer: for (int i = 0; i < forest.size() - 1; i++) { // C
			var tree = forest.get(i);
			var incidentTree = incidentEdgesForest.get(i);
			// Find first incident edge by priority
			// Priority : Any edge between two trees that are not linked is prioritized over
			// an edge between two already linked trees
			for (int u = 0; u < tree.n; u++) { // M
				if (incidentTree.getDegree(u) == 0)
					continue;
				var v = incidentTree.edgeIterator(u).next();
				for (int j = 0; j < forest.size(); j++) { // C
					if (j == i)
						continue;
					if (forest.get(j).getDegree(v) > 0) {
						if (!treatedTrees[j].get(i)) {
							treatedTrees[i].flip(j);
							treatedTrees[j].flip(i);
							incidentEdgesGraph.addEdge(i, j);
							incidentEdgesValues[i].putIfAbsent(j, new Integer[] { v, u });
							incidentEdgesValues[j].putIfAbsent(i, new Integer[] { u, v });
							continue outer;
						}
					} 
				}
			}
		}

		var pathIncidentEdges = DFS(incidentEdgesGraph, 0); // V+E

		return new SpanningPath(pathIncidentEdges, incidentEdgesValues);
	}

	/**
	 * Add the edges to link the trees of the forest
	 * @param map
	 */
	private void addLinkingEdges(SpanningPath map) {
		for (int i = 0; i < map.path.n; i++) { // N
			var iterator = map.path.edgeIterator(i);
			while (iterator.hasNext()) { // +M
				var j = iterator.next();
				addEdge(map.values[i].get(j)[0], map.values[i].get(j)[1]);
			}
		}
	}

	private static record SpanningPath(Graph path, Map<Integer, Integer[]>[] values) {
	}
}
