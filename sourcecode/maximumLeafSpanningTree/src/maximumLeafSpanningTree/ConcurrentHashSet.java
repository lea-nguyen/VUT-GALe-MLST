package maximumLeafSpanningTree;

import java.util.HashSet;
import java.util.Set;

public class ConcurrentHashSet {
	private final HashSet<Integer> subset;
	private final Object lock = new Object();

	public ConcurrentHashSet() {
		subset =  new HashSet<Integer>();
	}


	public Set<Integer> get() {
		synchronized (lock) {
			return new HashSet<>(subset);
		}
	}

	public void add(int value) {
		synchronized (lock) {
			subset.add(value);
		}
	}

	public void addAll(Set<Integer> set2) {
		synchronized (lock) {
			subset.addAll(set2);
		}
	}

	public void print(int index) {
		synchronized (lock) {
			System.out.println(subset);
		}
	}

}
