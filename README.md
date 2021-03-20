## Graph Definition Language (GDL)

Inspired by the [Neo4j Cypher](http://neo4j.com/docs/stable/cypher-query-lang.html) query language, GDL allows the simple definition of property graphs.
GDL contains a parser and simple structs that represent the property graph and its elements.
The Rust implementation is inspired by my [Java implementation](https://github.com/s1ck/gdl).

### Property graph data model

A property graph consists of nodes and relationships.
Nodes have zero or more labels, relationships have zero or one relationship type.
Both, nodes and relationships have properties, organized as key-value-pairs.
Relationships are directed, starting at a source node and pointing at a target node.

### Quickstart example

```rust
use gdl::{CypherValue, Graph};
use std::rc::Rc;

let gdl_string = "(alice:Person { age: 23 }),
                  (bob:Person { age: 42 }),
                  (alice)-[r:KNOWS { since: 1984 }]->(bob)";

let graph = Graph::from(gdl_string).unwrap();

assert_eq!(graph.node_count(), 2);
assert_eq!(graph.relationship_count(), 1);

let alice = graph.get_node("alice").unwrap();
assert_eq!(alice.properties.get("age"), Some(&CypherValue::Integer(23)));

let relationship = graph.get_relationship("r").unwrap();
assert_eq!(relationship.rel_type, Some(Rc::new(String::from("KNOWS"))));
```

### More GDL language examples

Define a node:

```rust
let g = gdl::Graph::from("()").unwrap();

assert_eq!(g.node_count(), 1);
```

Define a node and assign it to variable `alice`:

```rust
let g = gdl::Graph::from("(alice)").unwrap();

assert!(g.get_node("alice").is_some());
```

Define a node with label `User` and a single property:

```rust
let g = gdl::Graph::from("(alice:User { age : 23 })").unwrap();

assert_eq!(g.get_node("alice").unwrap().labels.len(), 1);
assert!(g.get_node("alice").unwrap().properties.get("age").is_some());
```

 Define an outgoing relationship:

```rust
let g = gdl::Graph::from("(alice)-->()").unwrap();

assert_eq!(g.relationship_count(), 1);
```

Define an incoming relationship:

```rust
let g = gdl::Graph::from("(alice)<--()").unwrap();

assert_eq!(g.relationship_count(), 1);
```

Define a relationship with type `KNOWS`, assign it to variable `r1` and add a property:

```rust
use std::rc::Rc;

let g = gdl::Graph::from("(alice)-[r1:KNOWS { since : 2014 }]->(bob)").unwrap();

assert!(g.get_relationship("r1").is_some());
assert_eq!(g.get_relationship("r1").unwrap().rel_type, Some(Rc::new(String::from("KNOWS"))));
```

Define multiple outgoing relationships from the same source node (i.e. `alice`):

```rust
let g = gdl::Graph::from("
    (alice)-[r1:KNOWS { since : 2014 }]->(bob)
    (alice)-[r2:KNOWS { since : 2013 }]->(eve)
").unwrap();

assert_eq!(g.node_count(), 3);
assert_eq!(g.relationship_count(), 2);
```

Define paths (four nodes and three relationships are created):

```rust
let g = gdl::Graph::from("()-->()<--()-->()").unwrap();

assert_eq!(g.node_count(), 4);
assert_eq!(g.relationship_count(), 3);
```

Paths can be comma separated to express arbitrary complex patterns:

```rust
let g = gdl::Graph::from("
    ()-->()<--()-->(),
    ()<--()-->()-->(),
    ()-->()<--()-->()
").unwrap();

assert_eq!(g.node_count(), 12);
assert_eq!(g.relationship_count(), 9);
```

### License

Apache 2.0 or MIT
