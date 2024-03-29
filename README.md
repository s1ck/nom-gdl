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

let gdl_string = "(alice:Person { name: 'Alice', age: 23 }),
                  (bob:Person { name: 'Bob', age: 42 }),
                  (alice)-[r:KNOWS { since: 1984 }]->(bob)";

let graph = gdl_string.parse::<gdl::Graph>().unwrap();

assert_eq!(graph.node_count(), 2);
assert_eq!(graph.relationship_count(), 1);

let alice = graph.get_node("alice").unwrap();
assert_eq!(alice.property_value("age"), Some(&CypherValue::from(23)));
assert_eq!(alice.property_value("name"), Some(&CypherValue::from("Alice")));

let relationship = graph.get_relationship("r").unwrap();
assert_eq!(relationship.rel_type(), Some("KNOWS"));
```

### More GDL language examples

Define a node:

```rust
let g = "()".parse::<gdl::Graph>().unwrap();

assert_eq!(g.node_count(), 1);
```

Define a node and assign it to variable `alice`:

```rust
let g = "(alice)".parse::<gdl::Graph>().unwrap();

assert!(g.get_node("alice").is_some());
```

Define a node with label `User` and multiple properties:

```rust
let g = "(alice:User { name: 'Alice', age : 23 })".parse::<gdl::Graph>().unwrap();

assert_eq!(g.get_node("alice").unwrap().labels().collect::<Vec<_>>(), vec!["User"]);
assert!(g.get_node("alice").unwrap().property_value("name").is_some());
assert!(g.get_node("alice").unwrap().property_value("age").is_some());
```

 Define an outgoing relationship:

```rust
let g = "(alice)-->()".parse::<gdl::Graph>().unwrap();

assert_eq!(g.relationship_count(), 1);
```

Define an incoming relationship:

```rust
let g = "(alice)<--()".parse::<gdl::Graph>().unwrap();

assert_eq!(g.relationship_count(), 1);
```

Define a relationship with type `KNOWS`, assign it to variable `r1` and add a property:

```rust
use std::rc::Rc;

let g = "(alice)-[r1:KNOWS { since : 2014 }]->(bob)".parse::<gdl::Graph>().unwrap();

assert!(g.get_relationship("r1").is_some());
assert_eq!(g.get_relationship("r1").unwrap().rel_type(), Some("KNOWS"));
```

Define multiple outgoing relationships from the same source node (i.e. `alice`):

```rust
let g = "
    (alice)-[r1:KNOWS { since : 2014 }]->(bob)
    (alice)-[r2:KNOWS { since : 2013 }]->(eve)
".parse::<gdl::Graph>().unwrap();

assert_eq!(g.node_count(), 3);
assert_eq!(g.relationship_count(), 2);
```

Define paths (four nodes and three relationships are created):

```rust
let g = "()-->()<--()-->()".parse::<gdl::Graph>().unwrap();

assert_eq!(g.node_count(), 4);
assert_eq!(g.relationship_count(), 3);
```

Paths can be comma separated to express arbitrary complex patterns:

```rust
let g = "
    ()-->()<--()-->(),
    ()<--()-->()-->(),
    ()-->()<--()-->()
".parse::<gdl::Graph>().unwrap();

assert_eq!(g.node_count(), 12);
assert_eq!(g.relationship_count(), 9);
```

### License

Apache 2.0 or MIT

