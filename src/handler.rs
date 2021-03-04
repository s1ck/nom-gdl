use std::{
    collections::{hash_map::Entry, HashMap},
    rc::Rc,
};

use thiserror::Error;

use crate::parser::{
    Direction, Graph as ParseGraph, Node as ParseNode, Path as ParsePath,
    Relationship as ParseRelationship,
};

#[derive(Error, Debug, PartialEq, Eq)]
pub enum GraphHandlerError {
    #[error("multiple declaration of variable `{0}`")]
    MultipleDeclarations(String),
}

#[derive(PartialEq, Eq, Debug)]
pub struct Node {
    id: usize,
    identifier: String,
    labels: Vec<Rc<String>>,
}

impl Node {
    fn new(id: usize, identifier: &str, labels: Vec<impl Into<String>>) -> Self {
        Self {
            id,
            identifier: identifier.to_string(),
            labels: labels
                .into_iter()
                .map(|label| Rc::new(label.into()))
                .collect(),
        }
    }
}
#[derive(PartialEq, Eq, Debug, Default)]
pub struct Relationship {
    id: usize,
    source_id: usize,
    target_id: usize,
    identifier: String,
    rel_type: Option<Rc<String>>,
}

impl Relationship {
    #[cfg(test)]
    fn new(identifier: impl Into<String>, rel_type: Option<&str>) -> Self {
        Self {
            identifier: identifier.into(),
            rel_type: rel_type.map(|s| Rc::new(s.to_string())),
            ..Relationship::default()
        }
    }
}

#[derive(Default)]
struct GraphHandler {
    token_cache: HashMap<String, Rc<String>>,
    node_cache: HashMap<String, Node>,
    relationship_cache: HashMap<String, Relationship>,
}

impl GraphHandler {
    fn node(&mut self, parse_node: ParseNode) -> Result<&Node, GraphHandlerError> {
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
                };

                Ok(entry.insert(new_node))
            }
        }
    }

    fn relationship(
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
                if parse_relationship.rel_type.is_some() {
                    return Err(GraphHandlerError::MultipleDeclarations(entry.key().clone()));
                }
                Ok(entry.into_mut())
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
                };

                Ok(entry.insert(new_relationship))
            }
        }
    }

    fn path(&mut self, parse_path: ParsePath) -> Result<(), GraphHandlerError> {
        let mut first_node_id = self.node(parse_path.start)?.id;

        for (parse_rel, parse_node) in parse_path.elements.into_iter() {
            let direction = parse_rel.direction;
            let second_node_id = self.node(parse_node)?.id;
            let relationship = self.relationship(parse_rel)?;

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

    fn graph(&mut self, parse_graph: ParseGraph) -> Result<(), GraphHandlerError> {
        for parse_path in parse_graph.paths {
            self.path(parse_path)?
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_case::test_case;

    #[test_case("()", Node::new(0, "__v0", Vec::<String>::new()) ; "empty")]
    #[test_case("(a)", Node::new(0, "a", Vec::<String>::new()) ; "identifier only")]
    #[test_case("(:A)", Node::new(0, "__v0", vec!["A"]) ; "label only")]
    #[test_case("(a:A)", Node::new(0, "a", vec!["A"]) ; "full")]
    fn convert_parse_node(input: &str, expected: Node) {
        let parse_node = input.parse::<ParseNode>().unwrap();
        let mut graph_handler = GraphHandler::default();
        let node = graph_handler.node(parse_node).unwrap();

        assert_eq!(*node, expected)
    }

    #[test_case("-->", Relationship::new("__r0", None) ; "empty")]
    #[test_case("-[r]->", Relationship::new("r", None) ; "identifier only")]
    #[test_case("-[:R]->", Relationship::new("__r0", Some("R")) ; "rel_type only")]
    #[test_case("-[r:R]->", Relationship::new("r", Some("R")) ; "full")]
    fn convert_parse_relationship(input: &str, expected: Relationship) {
        let parse_relationship = input.parse::<ParseRelationship>().unwrap();
        let mut graph_handler = GraphHandler::default();
        let relationship = graph_handler.relationship(parse_relationship).unwrap();

        assert_eq!(*relationship, expected)
    }

    #[test]
    fn convert_path() {
        let parse_path = "(a)-[r1]->(b)<-[r2]-(a)".parse::<ParsePath>().unwrap();
        let mut graph_handler = GraphHandler::default();
        graph_handler.path(parse_path).unwrap();

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
        let mut graph_handler = GraphHandler::default();
        graph_handler.graph(parse_graph).unwrap();

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
    fn multiple_declarations_error() {
        let parse_path = "(a:A)-->(a:B)".parse::<ParsePath>().unwrap();
        let mut graph_handler = GraphHandler::default();
        let error = graph_handler.path(parse_path).unwrap_err();

        assert_eq!(
            error,
            GraphHandlerError::MultipleDeclarations("a".to_string())
        );
    }
}
