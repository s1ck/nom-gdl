use std::{
    collections::{hash_map::Entry, HashMap},
    rc::Rc,
};

use thiserror::Error;

use crate::parser::{
    CypherValue, Direction, Graph as ParseGraph, Node as ParseNode, Path as ParsePath,
    Relationship as ParseRelationship,
};

#[derive(Error, Debug, PartialEq)]
pub enum GraphHandlerError {
    #[error("multiple declaration of node variable `{0}`")]
    MultipleDeclarations(String),
    #[error("invalid reference of relationship variable `{0}`")]
    InvalidReference(String),
    #[error("error during parsing")]
    Parser(#[from] nom::error::Error<String>),
}

#[derive(PartialEq, Debug, Default)]
pub struct Node {
    id: usize,
    variable: String,
    labels: Vec<Rc<String>>,
    properties: HashMap<String, CypherValue>,
}

impl Node {
    pub fn id(&self) -> usize {
        self.id
    }

    pub fn variable(&self) -> &str {
        self.variable.as_str()
    }

    pub fn labels(&self) -> impl Iterator<Item = &str> {
        self.labels.iter().map(|label| (*label).as_str())
    }

    pub fn property_keys(&self) -> impl Iterator<Item = &str> {
        self.properties.keys().map(|k| k.as_str())
    }

    pub fn property_value(&self, key: &str) -> Option<&CypherValue> {
        self.properties.get(key)
    }

    pub fn properties(&self) -> impl Iterator<Item = (&str, &CypherValue)> {
        self.properties.iter().map(|(k, v)| (k.as_str(), v))
    }
}

#[derive(PartialEq, Debug, Default)]
pub struct Relationship {
    id: usize,
    source_id: usize,
    target_id: usize,
    variable: String,
    rel_type: Option<Rc<String>>,
    properties: HashMap<String, CypherValue>,
}

impl Relationship {
    pub fn id(&self) -> usize {
        self.id
    }

    pub fn source_id(&self) -> usize {
        self.source_id
    }

    pub fn target_id(&self) -> usize {
        self.target_id
    }

    pub fn variable(&self) -> &str {
        self.variable.as_str()
    }

    pub fn rel_type(&self) -> Option<&str> {
        self.rel_type.as_ref().map(|rel_type| (**rel_type).as_str())
    }

    pub fn property_keys(&self) -> impl Iterator<Item = &str> {
        self.properties.keys().map(|k| k.as_str())
    }

    pub fn property_value(&self, key: &str) -> Option<&CypherValue> {
        self.properties.get(key)
    }

    pub fn properties(&self) -> impl Iterator<Item = (&str, &CypherValue)> {
        self.properties.iter().map(|(k, v)| (k.as_str(), v))
    }
}

#[derive(Default)]
pub struct Graph {
    token_cache: HashMap<String, Rc<String>>,
    node_cache: HashMap<String, Node>,
    relationship_cache: HashMap<String, Relationship>,
}

impl Graph {
    /// Creates a new graph from the given GDL string.
    ///
    /// Example
    ///
    /// ```
    /// use gdl::Graph;
    /// use gdl::CypherValue;
    /// use std::rc::Rc;
    ///
    /// let graph = Graph::from(
    ///     "(alice:Person { age: 23 }),
    ///     (bob:Person { age: 42 }),
    ///     (alice)-[r:KNOWS]->(bob)",
    ///     )
    ///     .unwrap();
    ///
    /// assert_eq!(graph.node_count(), 2);
    /// assert_eq!(graph.relationship_count(), 1);
    ///
    /// let alice = graph.get_node("alice").unwrap();
    ///
    /// assert_eq!(alice.property_value("age"), Some(&CypherValue::from(23)));
    ///
    /// let relationship = graph.get_relationship("r").unwrap();
    /// assert_eq!(relationship.rel_type(), Some("KNOWS"));
    /// ```
    pub fn from(input: &str) -> Result<Self, GraphHandlerError> {
        let mut graph_handler = Self::default();
        graph_handler.append(input)?;
        Ok(graph_handler)
    }

    /// Parses the given GDL string and updates the graph state.
    ///
    /// Example
    ///
    /// ```
    /// use gdl::Graph;
    ///
    /// let mut graph = Graph::from("(alice),(bob)").unwrap();
    ///
    /// graph.append("(alice)-[:KNOWS]->(bob)");
    /// graph.append("(bob)-[:KNOWS]->(eve)");
    ///
    /// assert_eq!(graph.node_count(), 3);
    /// assert_eq!(graph.relationship_count(), 2);
    /// ```
    pub fn append(&mut self, input: &str) -> Result<(), GraphHandlerError> {
        let parse_graph = input.parse::<ParseGraph>()?;
        self.convert_graph(parse_graph)?;
        Ok(())
    }

    /// Returns the number of nodes in the graph.
    ///
    /// Example
    ///
    /// ```
    /// use gdl::Graph;
    ///
    /// let graph = Graph::from("()-->()-->()-->()").unwrap();
    ///
    /// assert_eq!(graph.node_count(), 4);
    /// ```
    pub fn node_count(&self) -> usize {
        self.node_cache.len()
    }

    /// Returns the number of relationships in the graph.
    ///
    /// Example
    ///
    /// ```
    /// use gdl::Graph;
    ///
    /// let graph = Graph::from("()-->()-->()-->()").unwrap();
    ///
    /// assert_eq!(graph.relationship_count(), 3);
    /// ```
    pub fn relationship_count(&self) -> usize {
        self.relationship_cache.len()
    }

    /// Returns the node for the given variable.
    ///
    /// Example:
    ///
    /// ```
    /// use gdl::Graph;
    /// use gdl::CypherValue;
    /// use std::rc::Rc;
    ///
    /// let graph = Graph::from("(n0:A:B { foo: 42, bar: 1337 })").unwrap();
    ///
    /// let n0 = graph.get_node("n0").unwrap();
    ///
    /// assert_eq!(n0.variable(), String::from("n0"));
    /// assert_eq!(n0.labels().collect::<Vec<_>>(), vec!["A", "B"]);
    /// assert_eq!(n0.property_value("foo").unwrap(), &CypherValue::from(42));
    /// ```
    pub fn get_node(&self, variable: &str) -> Option<&Node> {
        self.node_cache.get(variable)
    }

    /// Returns the relationship for the given variable.
    ///
    /// Example:
    ///
    /// ```
    /// use gdl::Graph;
    /// use gdl::CypherValue;
    /// use std::rc::Rc;
    ///
    /// let graph = Graph::from("()-[r0:REL { foo: 42, bar: 13.37 }]->()").unwrap();
    ///
    /// let r0 = graph.get_relationship("r0").unwrap();
    ///
    /// assert_eq!(r0.variable(), String::from("r0"));
    /// assert_eq!(r0.rel_type(), Some("REL"));
    /// assert_eq!(r0.property_value("bar").unwrap(), &CypherValue::from(13.37));
    /// ```
    pub fn get_relationship(&self, variable: &str) -> Option<&Relationship> {
        self.relationship_cache.get(variable)
    }

    /// Returns an iterator of nodes contained in the graph.
    ///
    /// Example
    ///
    /// ```
    /// use gdl::Graph;
    ///
    /// let graph = Graph::from("(a),(b),(c)").unwrap();
    ///
    /// for node in graph.nodes() {
    ///     println!("{:?}", node);
    /// }
    /// ```
    pub fn nodes(&self) -> impl Iterator<Item = &Node> {
        self.node_cache.values()
    }

    /// Returns an iterator of relationships contained in the graph.
    ///
    /// Example
    ///
    /// ```
    /// use gdl::Graph;
    ///
    /// let graph = Graph::from("(a)-->(b)-->(c)").unwrap();
    ///
    /// for relationship in graph.relationships() {
    ///     println!("{:?}", relationship);
    /// }
    /// ```
    pub fn relationships(&self) -> impl Iterator<Item = &Relationship> {
        self.relationship_cache.values()
    }
}

impl Graph {
    fn convert_node(&mut self, parse_node: ParseNode) -> Result<&Node, GraphHandlerError> {
        // if the node is not in the cache, we
        // use the next_id as node id and variable
        let next_id = self.node_cache.len();

        let variable = match parse_node.variable {
            Some(variable) => variable,
            None => format!("__v{}", next_id),
        };

        let token_cache = &mut self.token_cache;

        match self.node_cache.entry(variable) {
            Entry::Occupied(entry) => {
                // verify that parse node has no additional content
                if parse_node.labels.len() > 0 {
                    return Err(GraphHandlerError::MultipleDeclarations(entry.key().clone()));
                }
                Ok(entry.into_mut())
            }
            Entry::Vacant(entry) => {
                let variable = entry.key().clone();
                let labels = parse_node
                    .labels
                    .into_iter()
                    .map(|label| match token_cache.entry(label) {
                        Entry::Occupied(entry) => Rc::clone(entry.get()),
                        Entry::Vacant(entry) => {
                            let label = entry.key().clone();
                            Rc::clone(entry.insert(Rc::new(label)))
                        }
                    })
                    .collect();

                let new_node = Node {
                    id: next_id,
                    variable,
                    labels,
                    properties: parse_node.properties,
                };

                Ok(entry.insert(new_node))
            }
        }
    }

    fn convert_relationship(
        &mut self,
        parse_relationship: ParseRelationship,
    ) -> Result<&mut Relationship, GraphHandlerError> {
        let next_id = self.relationship_cache.len();

        let variable = match parse_relationship.variable {
            Some(variable) => variable,
            None => format!("__r{}", next_id),
        };

        let token_cache = &mut self.token_cache;

        match self.relationship_cache.entry(variable) {
            Entry::Occupied(entry) => {
                // Relationships can not be referenced multiple times.
                Err(GraphHandlerError::InvalidReference(entry.key().clone()))
            }
            Entry::Vacant(entry) => {
                let variable = entry.key().clone();

                let rel_type =
                    parse_relationship
                        .rel_type
                        .map(|rel_type| match token_cache.entry(rel_type) {
                            Entry::Occupied(entry) => Rc::clone(entry.get()),
                            Entry::Vacant(entry) => {
                                let rel_type = entry.key().clone();
                                Rc::clone(entry.insert(Rc::new(rel_type)))
                            }
                        });

                let new_relationship = Relationship {
                    id: next_id,
                    source_id: usize::default(),
                    target_id: usize::default(),
                    variable,
                    rel_type,
                    properties: parse_relationship.properties,
                };

                Ok(entry.insert(new_relationship))
            }
        }
    }

    fn convert_path(&mut self, parse_path: ParsePath) -> Result<(), GraphHandlerError> {
        let mut first_node_id = self.convert_node(parse_path.start)?.id;

        for (parse_rel, parse_node) in parse_path.elements.into_iter() {
            let direction = parse_rel.direction;
            let second_node_id = self.convert_node(parse_node)?.id;
            let relationship = self.convert_relationship(parse_rel)?;

            match direction {
                Direction::Outgoing => {
                    relationship.source_id = first_node_id;
                    relationship.target_id = second_node_id;
                }
                Direction::Incoming => {
                    relationship.source_id = second_node_id;
                    relationship.target_id = first_node_id;
                }
            }

            first_node_id = second_node_id;
        }
        Ok(())
    }

    fn convert_graph(&mut self, parse_graph: ParseGraph) -> Result<(), GraphHandlerError> {
        for parse_path in parse_graph.paths {
            self.convert_path(parse_path)?
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nom::error::ErrorKind;
    use test_case::test_case;

    impl Node {
        fn new(
            variable: &str,
            labels: Vec<impl Into<String>>,
            properties: HashMap<impl Into<String>, CypherValue>,
        ) -> Self {
            Self {
                variable: variable.to_string(),
                labels: labels
                    .into_iter()
                    .map(|label| Rc::new(label.into()))
                    .collect(),
                properties: properties
                    .into_iter()
                    .map(|(k, v)| (Into::into(k), v))
                    .collect(),
                ..Node::default()
            }
        }
    }

    impl Relationship {
        fn new<T>(variable: T, rel_type: Option<&str>, properties: HashMap<T, CypherValue>) -> Self
        where
            T: Into<String>,
        {
            Self {
                variable: variable.into(),
                rel_type: rel_type.map(|s| Rc::new(s.to_string())),
                properties: properties
                    .into_iter()
                    .map(|(k, v)| (Into::into(k), v))
                    .collect::<HashMap<_, _>>(),
                ..Relationship::default()
            }
        }
    }

    #[test_case("()", Node::new("__v0", Vec::<String>::new(), HashMap::<String, CypherValue>::new()) ; "empty")]
    #[test_case("(a)", Node::new("a", Vec::<String>::new(), HashMap::<String, CypherValue>::new()) ; "variable only")]
    #[test_case("(:A)", Node::new("__v0", vec!["A"], HashMap::<String, CypherValue>::new()) ; "label only")]
    #[test_case("(a:A)", Node::new("a", vec!["A"], HashMap::<String, CypherValue>::new()) ; "variable and label")]
    #[test_case("(a:A { foo: 42, bar: 'foobar' })", Node::new("a", vec!["A"], vec![("foo", CypherValue::from(42)), ("bar", CypherValue::from("foobar"))].into_iter().collect::<HashMap<_,_>>()); "full")]
    fn convert_node(input: &str, expected: Node) {
        let parse_node = input.parse::<ParseNode>().unwrap();
        let mut graph_handler = Graph::default();
        let node = graph_handler.convert_node(parse_node).unwrap();

        assert_eq!(*node, expected)
    }

    #[test_case("-->", Relationship::new("__r0", None, HashMap::default()) ; "empty")]
    #[test_case("-[r]->", Relationship::new("r", None, HashMap::default()) ; "variable only")]
    #[test_case("-[:R]->", Relationship::new("__r0", Some("R"), HashMap::default()) ; "rel type only")]
    #[test_case("-[r:R]->", Relationship::new("r", Some("R"), HashMap::default()) ; "variable and rel type")]
    #[test_case("-[r:R { foo: 42 }]->", Relationship::new("r", Some("R"), std::iter::once(("foo", CypherValue::from(42))).collect::<HashMap<_,_>>()) ; "full")]
    fn convert_relationship(input: &str, expected: Relationship) {
        let parse_relationship = input.parse::<ParseRelationship>().unwrap();
        let mut graph_handler = Graph::default();
        let relationship = graph_handler
            .convert_relationship(parse_relationship)
            .unwrap();

        assert_eq!(*relationship, expected)
    }

    #[test]
    fn convert_path() {
        let parse_path = "(a)-[r1]->(b)<-[r2]-(a)".parse::<ParsePath>().unwrap();
        let mut graph_handler = Graph::default();
        graph_handler.convert_path(parse_path).unwrap();

        let node_a = graph_handler.node_cache.get("a").unwrap();
        let node_b = graph_handler.node_cache.get("b").unwrap();
        let rel_r1 = graph_handler.relationship_cache.get("r1").unwrap();
        let rel_r2 = graph_handler.relationship_cache.get("r2").unwrap();

        assert_eq!(node_a.id, rel_r1.source_id);
        assert_eq!(node_b.id, rel_r1.target_id);
        assert_eq!(node_b.id, rel_r2.target_id);
        assert_eq!(node_a.id, rel_r1.source_id);
    }

    #[test]
    fn convert_graph() {
        let parse_graph = "(a)-[r1]->(b),(b)<-[r2]-(a)".parse::<ParseGraph>().unwrap();
        let mut graph_handler = Graph::default();
        graph_handler.convert_graph(parse_graph).unwrap();

        let node_a = graph_handler.node_cache.get("a").unwrap();
        let node_b = graph_handler.node_cache.get("b").unwrap();
        let rel_r1 = graph_handler.relationship_cache.get("r1").unwrap();
        let rel_r2 = graph_handler.relationship_cache.get("r2").unwrap();

        assert_eq!(node_a.id, rel_r1.source_id);
        assert_eq!(node_b.id, rel_r1.target_id);
        assert_eq!(node_b.id, rel_r2.target_id);
        assert_eq!(node_a.id, rel_r1.source_id);
    }

    #[test]
    fn get_node() {
        let mut graph_handler = Graph::default();
        graph_handler.append("(n0),(n1),()").unwrap();

        assert_eq!(graph_handler.node_count(), 3);
        assert!(graph_handler.get_node("n0").is_some());
        assert!(graph_handler.get_node("n1").is_some());
    }

    #[test]
    fn get_relationship() {
        let mut graph_handler = Graph::default();
        graph_handler.append("()-->()-[r0]->()<-[r1]-()").unwrap();

        assert_eq!(graph_handler.relationship_count(), 3);
        assert!(graph_handler.get_relationship("r0").is_some());
        assert!(graph_handler.get_relationship("r1").is_some());
    }

    #[test]
    fn node_api() {
        let mut properties = HashMap::<String, CypherValue>::new();
        properties.insert("foo".to_string(), CypherValue::from(42));

        let n = Node {
            id: 42,
            variable: "n42".into(),
            labels: vec![Rc::new("A".into()), Rc::new("B".into())],
            properties,
        };

        assert_eq!(n.id(), 42);
        assert_eq!(n.variable(), "n42");
        assert_eq!(n.labels().collect::<Vec<_>>(), vec!["A", "B"]);
        assert_eq!(n.property_keys().collect::<Vec<_>>(), vec!["foo"]);
        assert_eq!(n.property_value("foo").unwrap(), &CypherValue::from(42));
        assert_eq!(
            n.properties().collect::<Vec<_>>(),
            vec![("foo", &CypherValue::from(42))]
        )
    }

    #[test]
    fn relationship_api() {
        let mut properties = HashMap::<String, CypherValue>::new();
        properties.insert("foo".into(), CypherValue::from(42));

        let r = Relationship {
            id: 42,
            source_id: 13,
            target_id: 37,
            variable: "r42".to_string(),
            rel_type: Some(Rc::new("REL".to_string())),
            properties,
        };

        assert_eq!(r.id(), 42);
        assert_eq!(r.source_id(), 13);
        assert_eq!(r.target_id(), 37);
        assert_eq!(r.variable(), "r42");
        assert_eq!(r.rel_type(), Some("REL"));
        assert_eq!(r.property_keys().collect::<Vec<_>>(), vec!["foo"]);
        assert_eq!(r.property_value("foo").unwrap(), &CypherValue::from(42));
        assert_eq!(
            r.properties().collect::<Vec<_>>(),
            vec![("foo", &CypherValue::from(42))]
        )
    }

    #[test]
    fn multiple_declarations_error() {
        let parse_path = "(a:A)-->(a:B)".parse::<ParsePath>().unwrap();
        let mut graph_handler = Graph::default();
        let error = graph_handler.convert_path(parse_path).unwrap_err();

        assert_eq!(
            error,
            GraphHandlerError::MultipleDeclarations("a".to_string())
        );
    }

    #[test]
    fn append_gdl() {
        let mut graph_handler = Graph::from("(a)").unwrap();
        graph_handler.append("(a)-->(b)").unwrap();
        graph_handler.append("(b)-->(c)").unwrap();

        assert_eq!(graph_handler.node_count(), 3);
        assert_eq!(graph_handler.relationship_count(), 2);
    }

    #[test]
    fn nodes_iterator() {
        let graph_handler = Graph::from("(a),(b),(c),(d),()").unwrap();
        let mut nodes = graph_handler
            .nodes()
            .map(|node| node.variable.as_str())
            .collect::<Vec<_>>();
        nodes.sort();
        assert_eq!(nodes, vec!["__v4", "a", "b", "c", "d"]);
    }

    #[test]
    fn relationships_iterator() {
        let graph_handler = Graph::from("()-[r1]->()-[r2]->()-->()").unwrap();
        let mut rels = graph_handler
            .relationships()
            .map(|rel| rel.variable.as_str())
            .collect::<Vec<_>>();
        rels.sort();
        assert_eq!(rels, vec!["__r2", "r1", "r2"]);
    }

    #[test]
    fn invalid_reference_error() {
        let parse_path = "(a)-[r1]->(b)-[r1]->(c)".parse::<ParsePath>().unwrap();
        let mut graph_handler = Graph::default();
        let error = graph_handler.convert_path(parse_path).unwrap_err();

        assert_eq!(error, GraphHandlerError::InvalidReference("r1".to_string()));
    }

    #[test]
    fn parser_error() {
        let mut graph_handler = Graph::default();
        let error = graph_handler.append("(a)-->(42:A)").unwrap_err();
        assert_eq!(
            error,
            GraphHandlerError::Parser(nom::error::Error::new("42:A)".to_string(), ErrorKind::Tag))
        )
    }
}
