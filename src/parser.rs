#![allow(dead_code)]

use std::str::FromStr;

use nom::{
    branch::alt,
    bytes::complete::{tag, take_while, take_while1},
    character::complete::{alpha1, alphanumeric1, multispace0},
    combinator::{map, opt, recognize},
    error::Error,
    multi::{many0, many1},
    sequence::{delimited, pair, preceded, terminated},
    Finish, IResult,
};
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Node {
    identifier: Option<String>,
    labels: Vec<String>,
}

impl Node {
    pub fn new(identifier: Option<String>, labels: Vec<String>) -> Self {
        Self { identifier, labels }
    }

    fn with_identifier(identifier: impl Into<String>) -> Self {
        Node {
            identifier: Some(identifier.into()),
            labels: vec![],
        }
    }

    fn with_identifier_and_labels<I, T>(identifier: impl Into<String>, labels: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<String>,
    {
        Node {
            identifier: Some(identifier.into()),
            labels: labels.into_iter().map(Into::into).collect(),
        }
    }
}

impl From<(Option<String>, Vec<String>)> for Node {
    fn from((identifier, labels): (Option<String>, Vec<String>)) -> Self {
        Node { identifier, labels }
    }
}

impl FromStr for Node {
    type Err = Error<String>;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match node(s).finish() {
            Ok((_remaining, node)) => Ok(node),
            Err(Error { input, code }) => Err(Error {
                input: input.to_string(),
                code,
            }),
        }
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct Relationship {
    identifier: Option<String>,
    rel_type: Option<String>,
    direction: Direction,
}
#[derive(Debug, PartialEq, Eq)]
enum Direction {
    Outgoing,
    Incoming,
}

impl Default for Direction {
    fn default() -> Self {
        Direction::Outgoing
    }
}

impl Relationship {
    fn outgoing(identifier: Option<String>, rel_type: Option<String>) -> Self {
        Relationship {
            identifier,
            rel_type,
            direction: Direction::Outgoing,
        }
    }

    fn outgoing_with_identifier(identifier: impl Into<String>) -> Self {
        Self::outgoing(Some(identifier.into()), None)
    }

    fn outgoing_with_identifier_and_rel_type(
        identifier: impl Into<String>,
        rel_type: impl Into<String>,
    ) -> Self {
        Self::outgoing(Some(identifier.into()), Some(rel_type.into()))
    }

    fn incoming(identifier: Option<String>, rel_type: Option<String>) -> Self {
        Relationship {
            identifier,
            rel_type,
            direction: Direction::Incoming,
        }
    }

    fn incoming_with_identifier(identifier: impl Into<String>) -> Self {
        Self::incoming(Some(identifier.into()), None)
    }

    fn incoming_with_identifier_and_rel_type(
        identifier: impl Into<String>,
        rel_type: impl Into<String>,
    ) -> Self {
        Self::incoming(Some(identifier.into()), Some(rel_type.into()))
    }
}

impl FromStr for Relationship {
    type Err = Error<String>;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match relationship(s).finish() {
            Ok((_remainder, relationship)) => Ok(relationship),
            Err(Error { input, code }) => Err(Error {
                input: input.to_string(),
                code,
            }),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Default)]
struct Path {
    start: Node,
    elements: Vec<(Relationship, Node)>,
}

impl Path {
    fn new(start: Node, elements: Vec<(Relationship, Node)>) -> Self {
        Self { start, elements }
    }
}

impl From<(Node, Vec<(Relationship, Node)>)> for Path {
    fn from((start, elements): (Node, Vec<(Relationship, Node)>)) -> Self {
        Self { start, elements }
    }
}

impl FromStr for Path {
    type Err = Error<String>;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match path(s).finish() {
            Ok((_remainder, path)) => Ok(path),
            Err(Error { input, code }) => Err(Error {
                input: input.to_string(),
                code,
            }),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Default)]
struct Graph {
    paths: Vec<Path>,
}

impl Graph {
    fn new(paths: Vec<Path>) -> Self {
        Self { paths }
    }
}

impl FromStr for Graph {
    type Err = Error<String>;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match graph(s).finish() {
            Ok((_remainder, graph)) => Ok(graph),
            Err(Error { input, code }) => Err(Error {
                input: input.to_string(),
                code,
            }),
        }
    }
}

fn is_uppercase_alphabetic(c: char) -> bool {
    c.is_alphabetic() && c.is_uppercase()
}

fn is_valid_label_token(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

fn is_valid_rel_type_token(c: char) -> bool {
    is_uppercase_alphabetic(c) || c.is_numeric() || c == '_'
}

fn identifier(input: &str) -> IResult<&str, String> {
    map(
        recognize(pair(
            alt((alpha1, tag("_"))),
            many0(alt((alphanumeric1, tag("_")))),
        )),
        String::from,
    )(input)
}

fn label(input: &str) -> IResult<&str, String> {
    map(
        preceded(
            tag(":"),
            recognize(pair(
                take_while1(is_uppercase_alphabetic),
                take_while(is_valid_label_token),
            )),
        ),
        String::from,
    )(input)
}

fn rel_type(input: &str) -> IResult<&str, String> {
    map(
        preceded(
            tag(":"),
            recognize(pair(
                take_while1(is_uppercase_alphabetic),
                take_while(is_valid_rel_type_token),
            )),
        ),
        String::from,
    )(input)
}

fn node_body(input: &str) -> IResult<&str, (Option<String>, Vec<String>)> {
    pair(opt(identifier), many0(label))(input)
}

fn node(input: &str) -> IResult<&str, Node> {
    map(delimited(tag("("), node_body, tag(")")), Node::from)(input)
}

fn relationship_body(input: &str) -> IResult<&str, (Option<String>, Option<String>)> {
    delimited(tag("["), pair(opt(identifier), opt(rel_type)), tag("]"))(input)
}

fn relationship(input: &str) -> IResult<&str, Relationship> {
    alt((
        map(
            delimited(tag("-"), opt(relationship_body), tag("->")),
            |relationship| match relationship {
                Some((identifier, rel_type)) => Relationship::outgoing(identifier, rel_type),
                None => Relationship::outgoing(None, None),
            },
        ),
        map(
            delimited(tag("<-"), opt(relationship_body), tag("-")),
            |relationship| match relationship {
                Some((identifier, rel_type)) => Relationship::incoming(identifier, rel_type),
                None => Relationship::incoming(None, None),
            },
        ),
    ))(input)
}

fn path(input: &str) -> IResult<&str, Path> {
    map(pair(node, many0(pair(relationship, node))), Path::from)(input)
}

fn graph(input: &str) -> IResult<&str, Graph> {
    map(
        many1(terminated(
            preceded(multispace0, path),
            preceded(multispace0, opt(tag(","))),
        )),
        Graph::new,
    )(input)
}

#[cfg(test)]
mod tests {
    use super::*;
    use parameterized::parameterized;
    use pretty_assertions::assert_eq as pretty_assert_eq;

    #[parameterized(
        input = {
            "foobar",
            "_foobar",
            "__foo_bar",
            "f",
            "F",
            "f1234",
        }
    )]
    fn identifiers_positive(input: &str) {
        let result = identifier(input);
        assert!(result.is_ok());
        let result = result.unwrap().1;
        assert_eq!(result, input)
    }

    #[parameterized(
        input = {
            "1234",
            "+foo",
            " ",
        }
    )]
    fn identifiers_negative(input: &str) {
        assert!(identifier(input).is_err())
    }

    #[parameterized(
        input = {
            ":Foobar",
            ":F",
            ":F42",
            ":F_42",
        }
    )]
    fn labels_positive(input: &str) {
        let result = label(input);
        assert!(result.is_ok());
        let result = result.unwrap().1;
        assert_eq!(format!(":{}", result), input)
    }

    #[parameterized(
        input = {
            ":foobar",
            ":_",
            "_",
        }
    )]
    fn labels_negative(input: &str) {
        assert!(label(input).is_err())
    }

    #[parameterized(
        input = {
            ":FOOBAR",
            ":F",
            ":F42",
            ":F_42",
        }
    )]
    fn rel_types_positive(input: &str) {
        let result = rel_type(input);
        assert!(result.is_ok());
        let result = result.unwrap().1;
        assert_eq!(format!(":{}", result), input)
    }

    #[test]
    fn node_empty() {
        assert_eq!("()".parse(), Ok(Node::default()))
    }

    #[test]
    fn node_with_identifier() {
        assert_eq!(
            "(n0)".parse(),
            Ok(Node {
                identifier: Some("n0".to_string()),
                ..Node::default()
            })
        );
    }
    #[test]
    fn node_with_labels() {
        assert_eq!(
            "(:A)".parse(),
            Ok(Node {
                labels: vec!["A".to_string()],
                ..Node::default()
            })
        );
        assert_eq!(
            "(:A:B)".parse(),
            Ok(Node {
                labels: vec!["A".to_string(), "B".to_string()],
                ..Node::default()
            })
        );
    }

    #[test]
    fn node_full() {
        assert_eq!(
            "(n0:A)".parse(),
            Ok(Node {
                identifier: Some("n0".to_string()),
                labels: vec!["A".to_string()],
            })
        );
        assert_eq!(
            "(n0:A:B)".parse(),
            Ok(Node {
                identifier: Some("n0".to_string()),
                labels: vec!["A".to_string(), "B".to_string()],
            })
        );
    }

    #[test]
    fn relationship_empty() {
        assert_eq!("-->".parse(), Ok(Relationship::outgoing(None, None)));
        assert_eq!("-[]->".parse(), Ok(Relationship::outgoing(None, None)));
        assert_eq!("<--".parse(), Ok(Relationship::incoming(None, None)));
        assert_eq!("<-[]-".parse(), Ok(Relationship::incoming(None, None)));
    }

    #[test]
    fn relationship_with_identifier() {
        assert_eq!(
            "-[r0]->".parse(),
            Ok(Relationship {
                identifier: Some("r0".to_string()),
                ..Relationship::default()
            })
        );
        assert_eq!(
            "<-[r0]-".parse(),
            Ok(Relationship {
                identifier: Some("r0".to_string()),
                direction: Direction::Incoming,
                ..Relationship::default()
            })
        );
    }

    #[test]
    fn relationship_with_rel_type() {
        assert_eq!(
            "-[:BAR]->".parse(),
            Ok(Relationship {
                rel_type: Some("BAR".to_string()),
                ..Relationship::default()
            })
        );
        assert_eq!(
            "<-[:BAR]-".parse(),
            Ok(Relationship {
                rel_type: Some("BAR".to_string()),
                direction: Direction::Incoming,
                ..Relationship::default()
            })
        );
    }

    #[test]
    fn relationship_full() {
        assert_eq!(
            "-[r0:BAR]->".parse(),
            Ok(Relationship {
                identifier: Some("r0".to_string()),
                rel_type: Some("BAR".to_string()),
                direction: Direction::Outgoing,
            })
        );
        assert_eq!(
            "<-[r0:BAR]-".parse(),
            Ok(Relationship {
                identifier: Some("r0".to_string()),
                rel_type: Some("BAR".to_string()),
                direction: Direction::Incoming,
            })
        )
    }

    #[test]
    fn path_node_only() {
        assert_eq!(
            "(a)".parse(),
            Ok(Path {
                start: Node::with_identifier("a"),
                elements: vec![]
            })
        );
    }

    #[test]
    fn path_one_hop_path() {
        assert_eq!(
            "(a)-->(b)".parse(),
            Ok(Path {
                start: Node::with_identifier("a"),
                elements: vec![(
                    Relationship::outgoing(None, None),
                    Node::with_identifier("b")
                )]
            })
        );
    }

    #[test]
    fn path_two_hop_path() {
        assert_eq!(
            "(a)-->(b)<--(c)".parse(),
            Ok(Path {
                start: Node::with_identifier("a"),
                elements: vec![
                    (
                        Relationship::outgoing(None, None),
                        Node::with_identifier("b")
                    ),
                    (
                        Relationship::incoming(None, None),
                        Node::with_identifier("c")
                    ),
                ]
            })
        );
    }

    #[test]
    fn path_with_node_labels_and_relationship_types() {
        assert_eq!(
            "(a:A)<-[:R]-(:B)-[rel]->(c)-[]->(d:D1:D2)<--(:E1:E2)-[r:REL]->()".parse(),
            Ok(Path {
                start: Node::with_identifier_and_labels("a", vec!["A"]),
                elements: vec![
                    (
                        Relationship::incoming(None, Some("R".to_string())),
                        Node::new(None, vec!["B".to_string()])
                    ),
                    (
                        Relationship::outgoing_with_identifier("rel"),
                        Node::with_identifier("c")
                    ),
                    (
                        Relationship::outgoing(None, None),
                        Node::with_identifier_and_labels("d", vec!["D1", "D2"])
                    ),
                    (
                        Relationship::incoming(None, None),
                        Node::new(None, vec!["E1".to_string(), "E2".to_string()])
                    ),
                    (
                        Relationship::outgoing_with_identifier_and_rel_type("r", "REL"),
                        Node::default()
                    ),
                ]
            })
        );
    }

    #[test]
    fn graph_one_paths() {
        pretty_assert_eq!(
            "(a)-->(b)".parse(),
            Ok(Graph::new(vec![Path {
                start: Node::with_identifier("a"),
                elements: vec![(
                    Relationship::outgoing(None, None),
                    Node::with_identifier("b")
                )]
            }]))
        );
    }

    #[parameterized(
        input = {
            "(a)(b)",
            "(a) (b)",
            "(a)  (b)",
            "(a),(b)",
            "(a), (b)",
            "(a) ,(b)",
            "(a) , (b)",
            "(a)  ,  (b)",
            "(a)  ,  (b),    ",
            r#"(a)
               (b)"#,
            r#"(a),
               (b)"#,
            r#"
              (a)
              (b)
            "#,
            r#"
              (a),
              (b)
            "#,
        }
    )]
    fn graph_two_paths(input: &str) {
        pretty_assert_eq!(
            input.parse(),
            Ok(Graph::new(vec![
                Path {
                    start: Node::with_identifier("a"),
                    elements: vec![]
                },
                Path {
                    start: Node::with_identifier("b"),
                    elements: vec![]
                }
            ]))
        );
    }

    #[test]
    fn graph_trailing_comma() {
        pretty_assert_eq!(
            "(a),".parse(),
            Ok(Graph::new(vec![Path {
                start: Node::with_identifier("a"),
                elements: vec![]
            }]))
        );
    }
}
