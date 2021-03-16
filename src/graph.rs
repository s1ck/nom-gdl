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

#[derive(PartialEq, Debug)]
pub struct Node {
    id: usize,
    identifier: String,
    labels: Vec<Rc<String>>,
    properties: HashMap<String, CypherValue>,
}

impl Node {
    fn new(
        id: usize,
        identifier: &str,
        labels: Vec<impl Into<String>>,
        properties: HashMap<impl Into<String>, CypherValue>,
    ) -> Self {
        Self {
            id,
            identifier: identifier.to_string(),
            labels: labels
                .into_iter()
                .map(|label| Rc::new(label.into()))
                .collect(),
            properties: properties
                .into_iter()
                .map(|(k, v)| (Into::into(k), v))
                .collect(),
        }
    }
}
#[derive(PartialEq, Debug, Default)]
pub struct Relationship {
    id: usize,
    source_id: usize,
    target_id: usize,
    identifier: String,
    rel_type: Option<Rc<String>>,
    properties: HashMap<String, CypherValue>,
}

impl Relationship {
    #[cfg(test)]
    fn new<T>(identifier: T, rel_type: Option<&str>, properties: HashMap<T, CypherValue>) -> Self
    where
        T: Into<String>,
    {
        Self {
            identifier: identifier.into(),
            rel_type: rel_type.map(|s| Rc::new(s.to_string())),
            properties: properties
                .into_iter()
                .map(|(k, v)| (Into::into(k), v))
                .collect::<HashMap<_, _>>(),
            ..Relationship::default()
        }
    }
}

#[derive(Default)]
pub struct GdlGraph {
    token_cache: HashMap<String, Rc<String>>,
    node_cache: HashMap<String, Node>,
    relationship_cache: HashMap<String, Relationship>,
}

impl GdlGraph {
    pub fn from(input: &str) -> Result<Self, GraphHandlerError> {
        let mut graph_handler = Self::default();
        graph_handler.parse(input)?;
        Ok(graph_handler)
    }

    pub fn parse(&mut self, input: &str) -> Result<(), GraphHandlerError> {
        let parse_graph = input.parse::<ParseGraph>()?;
        self.convert_graph(parse_graph)?;
        Ok(())
    }

    pub fn node_count(&self) -> usize {
        self.node_cache.len()
    }

    pub fn relationship_count(&self) -> usize {
        self.relationship_cache.len()
    }

    pub fn get_node(&self, identifier: &str) -> Option<&Node> {
        self.node_cache.get(identifier)
    }

    pub fn get_relationship(&self, identifier: &str) -> Option<&Relationship> {
        self.relationship_cache.get(identifier)
    }

    pub fn nodes(&self) -> impl Iterator<Item = &Node> {
        self.node_cache.values()
    }

    pub fn relationships(&self) -> impl Iterator<Item = &Relationship> {
        self.relationship_cache.values()
    }
}

impl GdlGraph {
    fn convert_node(&mut self, parse_node: ParseNode) -> Result<&Node, GraphHandlerError> {
        // if the node is not in the cache, we
        // use the next_id as node id and identifier
        let next_id = self.node_cache.len();

        let identifier = match parse_node.identifier {
            Some(identifier) => identifier,
            None => format!("__v{}", next_id),
        };

        let token_cache = &mut self.token_cache;

        match self.node_cache.entry(identifier) {
            Entry::Occupied(entry) => {
                // verify that parse node has no additional content
                if parse_node.labels.len() > 0 {
                    return Err(GraphHandlerError::MultipleDeclarations(entry.key().clone()));
                }
                Ok(entry.into_mut())
            }
            Entry::Vacant(entry) => {
                let identifier = entry.key().clone();
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
                    identifier,
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

        let identifier = match parse_relationship.identifier {
            Some(identifier) => identifier,
            None => format!("__r{}", next_id),
        };

        let token_cache = &mut self.token_cache;

        match self.relationship_cache.entry(identifier) {
            Entry::Occupied(entry) => {
                // Relationships can not be referenced multiple times.
                Err(GraphHandlerError::InvalidReference(entry.key().clone()))
            }
            Entry::Vacant(entry) => {
                let identifier = entry.key().clone();

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
                    identifier,
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

    #[test_case("()", Node::new(0, "__v0", Vec::<String>::new(), HashMap::<String, CypherValue>::new()) ; "empty")]
    #[test_case("(a)", Node::new(0, "a", Vec::<String>::new(), HashMap::<String, CypherValue>::new()) ; "identifier only")]
    #[test_case("(:A)", Node::new(0, "__v0", vec!["A"], HashMap::<String, CypherValue>::new()) ; "label only")]
    #[test_case("(a:A)", Node::new(0, "a", vec!["A"], HashMap::<String, CypherValue>::new()) ; "identifier and label")]
    #[test_case("(a:A { foo: 42 })", Node::new(0, "a", vec!["A"], std::iter::once(("foo", CypherValue::Integer(42))).collect::<HashMap<_,_>>()); "full")]
    fn convert_node(input: &str, expected: Node) {
        let parse_node = input.parse::<ParseNode>().unwrap();
        let mut graph_handler = GdlGraph::default();
        let node = graph_handler.convert_node(parse_node).unwrap();

        assert_eq!(*node, expected)
    }

    #[test_case("-->", Relationship::new("__r0", None, HashMap::default()) ; "empty")]
    #[test_case("-[r]->", Relationship::new("r", None, HashMap::default()) ; "identifier only")]
    #[test_case("-[:R]->", Relationship::new("__r0", Some("R"), HashMap::default()) ; "rel type only")]
    #[test_case("-[r:R]->", Relationship::new("r", Some("R"), HashMap::default()) ; "identifer and rel type")]
    #[test_case("-[r:R { foo: 42 }]->", Relationship::new("r", Some("R"), std::iter::once(("foo", CypherValue::Integer(42))).collect::<HashMap<_,_>>()) ; "full")]
    fn convert_relationship(input: &str, expected: Relationship) {
        let parse_relationship = input.parse::<ParseRelationship>().unwrap();
        let mut graph_handler = GdlGraph::default();
        let relationship = graph_handler
            .convert_relationship(parse_relationship)
            .unwrap();

        assert_eq!(*relationship, expected)
    }

    #[test]
    fn convert_path() {
        let parse_path = "(a)-[r1]->(b)<-[r2]-(a)".parse::<ParsePath>().unwrap();
        let mut graph_handler = GdlGraph::default();
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
        let mut graph_handler = GdlGraph::default();
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
        let mut graph_handler = GdlGraph::default();
        graph_handler.parse("(n0),(n1),()").unwrap();

        assert_eq!(graph_handler.node_count(), 3);
        assert!(graph_handler.get_node("n0").is_some());
        assert!(graph_handler.get_node("n1").is_some());
    }

    #[test]
    fn get_relationship() {
        let mut graph_handler = GdlGraph::default();
        graph_handler.parse("()-->()-[r0]->()<-[r1]-()").unwrap();

        assert_eq!(graph_handler.relationship_count(), 3);
        assert!(graph_handler.get_relationship("r0").is_some());
        assert!(graph_handler.get_relationship("r1").is_some());
    }

    #[test]
    fn multiple_declarations_error() {
        let parse_path = "(a:A)-->(a:B)".parse::<ParsePath>().unwrap();
        let mut graph_handler = GdlGraph::default();
        let error = graph_handler.convert_path(parse_path).unwrap_err();

        assert_eq!(
            error,
            GraphHandlerError::MultipleDeclarations("a".to_string())
        );
    }

    #[test]
    fn append_gdl() {
        let mut graph_handler = GdlGraph::from("(a)").unwrap();
        graph_handler.parse("(a)-->(b)").unwrap();
        graph_handler.parse("(b)-->(c)").unwrap();

        assert_eq!(graph_handler.node_count(), 3);
        assert_eq!(graph_handler.relationship_count(), 2);
    }

    #[test]
    fn nodes_iterator() {
        let graph_handler = GdlGraph::from("(a),(b),(c),(d),()").unwrap();
        let mut nodes = graph_handler
            .nodes()
            .map(|node| node.identifier.as_str())
            .collect::<Vec<_>>();
        nodes.sort();
        assert_eq!(nodes, vec!["__v4", "a", "b", "c", "d"]);
    }

    #[test]
    fn relationships_iterator() {
        let graph_handler = GdlGraph::from("()-[r1]->()-[r2]->()-->()").unwrap();
        let mut rels = graph_handler
            .relationships()
            .map(|rel| rel.identifier.as_str())
            .collect::<Vec<_>>();
        rels.sort();
        assert_eq!(rels, vec!["__r2", "r1", "r2"]);
    }

    #[test]
    fn invalid_reference_error() {
        let parse_path = "(a)-[r1]->(b)-[r1]->(c)".parse::<ParsePath>().unwrap();
        let mut graph_handler = GdlGraph::default();
        let error = graph_handler.convert_path(parse_path).unwrap_err();

        assert_eq!(error, GraphHandlerError::InvalidReference("r1".to_string()));
    }

    #[test]
    fn parser_error() {
        let mut graph_handler = GdlGraph::default();
        let error = graph_handler.parse("(a)-->(42:A)").unwrap_err();
        assert_eq!(
            error,
            GraphHandlerError::Parser(nom::error::Error::new("42:A)".to_string(), ErrorKind::Tag))
        )
    }
}
