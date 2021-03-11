use std::{collections::HashMap, str::FromStr};

use nom::character::complete::digit0;
use nom::{
    branch::alt,
    bytes::complete::{tag, take_while, take_while1},
    character::complete::{alpha1, alphanumeric1, digit1},
    combinator::{all_consuming, cut, map, opt, recognize},
    error::Error,
    multi::{many0, many1},
    sequence::{delimited, pair, preceded, terminated, tuple},
    Finish, IResult,
};
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct Node {
    pub(crate) identifier: Option<String>,
    pub(crate) labels: Vec<String>,
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
        match all_consuming(node)(s).finish() {
            Ok((_remaining, node)) => Ok(node),
            Err(Error { input, code }) => Err(Error {
                input: input.to_string(),
                code,
            }),
        }
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct Relationship {
    pub(crate) identifier: Option<String>,
    pub(crate) rel_type: Option<String>,
    pub(crate) direction: Direction,
}
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub(crate) enum Direction {
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
        match all_consuming(relationship)(s).finish() {
            Ok((_remainder, relationship)) => Ok(relationship),
            Err(Error { input, code }) => Err(Error {
                input: input.to_string(),
                code,
            }),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Default)]
pub(crate) struct Path {
    pub(crate) start: Node,
    pub(crate) elements: Vec<(Relationship, Node)>,
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
        match all_consuming(path)(s).finish() {
            Ok((_remainder, path)) => Ok(path),
            Err(Error { input, code }) => Err(Error {
                input: input.to_string(),
                code,
            }),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Default)]
pub(crate) struct Graph {
    pub(crate) paths: Vec<Path>,
}

impl Graph {
    fn new(paths: Vec<Path>) -> Self {
        Self { paths }
    }
}

impl FromStr for Graph {
    type Err = Error<String>;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match all_consuming(graph)(s).finish() {
            Ok((_remainder, graph)) => Ok(graph),
            Err(Error { input, code }) => Err(Error {
                input: input.to_string(),
                code,
            }),
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum CypherValue {
    Float(f64),
    Integer(i64),
}

impl FromStr for CypherValue {
    type Err = Error<String>;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match all_consuming(cypher_value)(s).finish() {
            Ok((_remainder, cypher_value)) => Ok(cypher_value),
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

fn sp(input: &str) -> IResult<&str, &str> {
    take_while(|c: char| c.is_ascii_whitespace())(input)
}

fn neg_sign(input: &str) -> IResult<&str, bool> {
    map(opt(tag("-")), |t| t.is_some())(input)
}

fn integer_literal(input: &str) -> IResult<&str, CypherValue> {
    map(pair(neg_sign, digit1), |(is_negative, num)| {
        let mut num = i64::from_str(num).unwrap();
        if is_negative {
            num = -num;
        }
        CypherValue::Integer(num)
    })(input)
}

fn float_literal(input: &str) -> IResult<&str, CypherValue> {
    map(
        pair(neg_sign, recognize(tuple((digit0, tag("."), digit0)))),
        |(is_negative, num)| {
            let mut num = f64::from_str(num).unwrap();
            if is_negative {
                num = -num;
            }
            CypherValue::Float(num)
        },
    )(input)
}

fn cypher_value(input: &str) -> IResult<&str, CypherValue> {
    preceded(sp, alt((float_literal, integer_literal)))(input)
}

fn identifier(input: &str) -> IResult<&str, String> {
    map(
        preceded(
            sp,
            recognize(pair(
                alt((alpha1, tag("_"))),
                many0(alt((alphanumeric1, tag("_")))),
            )),
        ),
        String::from,
    )(input)
}

fn key_value_pair(input: &str) -> IResult<&str, (String, CypherValue)> {
    pair(
        identifier,
        preceded(preceded(sp, tag(":")), cut(cypher_value)),
    )(input)
}

fn key_value_pairs(input: &str) -> IResult<&str, HashMap<String, CypherValue>> {
    map(
        pair(
            key_value_pair,
            many0(preceded(preceded(sp, tag(",")), key_value_pair)),
        ),
        |(head, tail)| std::iter::once(head).chain(tail).collect::<HashMap<_, _>>(),
    )(input)
}

fn properties(input: &str) -> IResult<&str, HashMap<String, CypherValue>> {
    map(
        delimited(
            preceded(sp, tag("{")),
            opt(key_value_pairs),
            preceded(sp, tag("}")),
        ),
        |properties| properties.unwrap_or_default(),
    )(input)
}

fn label(input: &str) -> IResult<&str, String> {
    map(
        preceded(
            tag(":"),
            cut(recognize(pair(
                take_while1(is_uppercase_alphabetic),
                take_while(is_valid_label_token),
            ))),
        ),
        String::from,
    )(input)
}

fn rel_type(input: &str) -> IResult<&str, String> {
    map(
        preceded(
            tag(":"),
            cut(recognize(pair(
                take_while1(is_uppercase_alphabetic),
                take_while(is_valid_rel_type_token),
            ))),
        ),
        String::from,
    )(input)
}

fn node_body(input: &str) -> IResult<&str, (Option<String>, Vec<String>)> {
    delimited(sp, pair(opt(identifier), many0(label)), sp)(input)
}

pub(crate) fn node(input: &str) -> IResult<&str, Node> {
    map(delimited(tag("("), node_body, tag(")")), Node::from)(input)
}

fn relationship_body(input: &str) -> IResult<&str, (Option<String>, Option<String>)> {
    delimited(
        tag("["),
        delimited(sp, pair(opt(identifier), opt(rel_type)), sp),
        tag("]"),
    )(input)
}

pub(crate) fn relationship(input: &str) -> IResult<&str, Relationship> {
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

pub(crate) fn path(input: &str) -> IResult<&str, Path> {
    map(pair(node, many0(pair(relationship, cut(node)))), Path::from)(input)
}

pub(crate) fn graph(input: &str) -> IResult<&str, Graph> {
    map(
        many1(terminated(preceded(sp, path), preceded(sp, opt(tag(","))))),
        Graph::new,
    )(input)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq as pretty_assert_eq;
    use test_case::test_case;

    #[test_case("0",     CypherValue::Integer(0)   ; "int: zero")]
    #[test_case("-0",    CypherValue::Integer(0)   ; "int: signed zero")]
    #[test_case("42",    CypherValue::Integer(42)  ; "int: positive")]
    #[test_case("-42",   CypherValue::Integer(-42) ; "int: negative")]
    #[test_case("0.0",   CypherValue::Float(0.0)   ; "float: zero v1")]
    #[test_case("0.",    CypherValue::Float(0.0)   ; "float: zero v2")]
    #[test_case(".0",    CypherValue::Float(0.0)   ; "float: zero v3")]
    #[test_case("-0.0",  CypherValue::Float(0.0)   ; "float: signed zero")]
    #[test_case("-.0",   CypherValue::Float(0.0)   ; "float: signed zero v2")]
    #[test_case("13.37", CypherValue::Float(13.37) ; "float: positive")]
    #[test_case("-42.2", CypherValue::Float(-42.2) ; "float: negative")]
    fn cypher_value(input: &str, expected: CypherValue) {
        assert_eq!(input.parse(), Ok(expected))
    }

    #[test_case("key:42",     ("key".to_string(), CypherValue::Integer(42)))]
    #[test_case("key: 1337",  ("key".to_string(), CypherValue::Integer(1337)))]
    #[test_case("key2: 1337", ("key2".to_string(), CypherValue::Integer(1337)))]
    fn key_value_pair_test(input: &str, expected: (String, CypherValue)) {
        assert_eq!(key_value_pair(input).unwrap().1, expected)
    }

    #[test_case("{key1: 42}", vec![("key1".to_string(), CypherValue::Integer(42))])]
    #[test_case("{key1: 13.37 }", vec![("key1".to_string(), CypherValue::Float(13.37))])]
    #[test_case("{ key1: 42, key2: 1337 }", vec![("key1".to_string(), CypherValue::Integer(42)), ("key2".to_string(), CypherValue::Integer(1337))])]
    fn properties_test(input: &str, expected: Vec<(String, CypherValue)>) {
        let expected = expected.into_iter().collect::<HashMap<_, _>>();
        assert_eq!(properties(input).unwrap().1, expected)
    }

    #[test_case("foobar"; "multiple alphabetical")]
    #[test_case("_foobar"; "starts with underscore")]
    #[test_case("__foo_bar"; "mixed underscore")]
    #[test_case("f"; "single alphabetical lowercase")]
    #[test_case("F"; "single alphabetical uppercase")]
    #[test_case("f1234"; "alphanumeric")]
    fn identifiers_positive(input: &str) {
        let result = identifier(input);
        assert!(result.is_ok());
        let result = result.unwrap().1;
        assert_eq!(result, input)
    }

    #[test_case("1234"; "numerical")]
    #[test_case("+foo"; "special char")]
    #[test_case("."; "another special char")]
    fn identifiers_negative(input: &str) {
        assert!(identifier(input).is_err())
    }

    #[test_case(":Foobar"; "alphabetical")]
    #[test_case(":F"; "alphabetical single char")]
    #[test_case(":F42"; "alphanumerical")]
    #[test_case(":F_42"; "alphanumerical and underscore")]
    fn labels_positive(input: &str) {
        let result = label(input);
        assert!(result.is_ok());
        let result = result.unwrap().1;
        assert_eq!(format!(":{}", result), input)
    }

    #[test_case(":foobar"; "lower case")]
    #[test_case(":_"; "colon and single underscore")]
    #[test_case("_"; "single underscore")]
    #[test_case(":1234"; "numerical")]
    fn labels_negative(input: &str) {
        assert!(label(input).is_err())
    }

    #[test_case(":FOOBAR"; "multiple alphabetical")]
    #[test_case(":F"; "single alphabetical")]
    #[test_case(":F42"; "alphanumerical")]
    #[test_case(":F_42"; "alphanumerical and underscore")]
    fn rel_types_positive(input: &str) {
        let result = rel_type(input);
        assert!(result.is_ok());
        let result = result.unwrap().1;
        assert_eq!(format!(":{}", result), input)
    }

    #[test_case("()"; "empty node")]
    #[test_case("( )"; "empty node with space")]
    #[test_case("(  )"; "empty node with many spaces")]
    fn node_empty(input: &str) {
        assert_eq!(input.parse(), Ok(Node::default()))
    }

    #[test_case("(n0)",   Node { identifier: Some("n0".to_string()), ..Node::default() }; "n0")]
    #[test_case("(n1)",   Node { identifier: Some("n1".to_string()), ..Node::default() }; "n1")]
    #[test_case("( n0 )", Node { identifier: Some("n0".to_string()), ..Node::default() }; "n0 with space")]
    fn node_with_identifier(input: &str, expected: Node) {
        assert_eq!(input.parse(), Ok(expected));
    }
    #[test_case("(:A)",     Node { labels: vec!["A".to_string()], ..Node::default() }; "single label" )]
    #[test_case("(:A:B)",   Node { labels: vec!["A".to_string(), "B".to_string()], ..Node::default() }; "multiple labels" )]
    #[test_case("( :A:B )", Node { labels: vec!["A".to_string(), "B".to_string()], ..Node::default() }; "multiple labels with space" )]
    fn node_with_labels(input: &str, expected: Node) {
        assert_eq!(input.parse(), Ok(expected));
    }

    #[test_case("(n0:A)",     Node { identifier: Some("n0".to_string()), labels: vec!["A".to_string()], }; "single label")]
    #[test_case("(n0:A:B)",   Node { identifier: Some("n0".to_string()), labels: vec!["A".to_string(), "B".to_string()], }; "multiple labels")]
    #[test_case("( n0:A:B )", Node { identifier: Some("n0".to_string()), labels: vec!["A".to_string(), "B".to_string()], }; "multiple labels with space")]
    fn node_full(input: &str, expected: Node) {
        assert_eq!(input.parse(), Ok(expected));
    }

    #[test_case("(42:A)" ; "numeric identifier")]
    #[test_case("("      ; "no closing")]
    #[test_case(")"      ; "no opening")]
    fn node_negative(input: &str) {
        assert!(input.parse::<Node>().is_err())
    }

    #[test_case("-->",    Relationship::outgoing(None, None); "outgoing: no body")]
    #[test_case("-[]->",  Relationship::outgoing(None, None); "outgoing: with body")]
    #[test_case("-[ ]->", Relationship::outgoing(None, None); "outgoing: body with space")]
    #[test_case("<--",    Relationship::incoming(None, None); "incoming: no body")]
    #[test_case("<-[]-",  Relationship::incoming(None, None); "incoming: with body")]
    #[test_case("<-[ ]-", Relationship::incoming(None, None); "incoming: body with space")]
    fn relationship_empty(input: &str, expected: Relationship) {
        assert_eq!(input.parse(), Ok(expected));
    }

    #[test_case("-[->" ; "outgoing: no closing")]
    #[test_case("-]->" ; "outgoing: no opening")]
    #[test_case("->"   ; "outgoing: no hyphen")]
    #[test_case("<-[-" ; "incoming: no closing")]
    #[test_case("<-]-" ; "incoming: no opening")]
    #[test_case("<-"   ; "incoming: no hyphen")]
    fn relationship_negative(input: &str) {
        assert!(input.parse::<Relationship>().is_err());
    }

    #[test_case("-[r0]->",   Relationship { identifier: Some("r0".to_string()), direction: Direction::Outgoing, ..Relationship::default() }; "outgoing")]
    #[test_case("-[ r0 ]->", Relationship { identifier: Some("r0".to_string()), direction: Direction::Outgoing, ..Relationship::default() }; "outgoing with space")]
    #[test_case("<-[r0]-",   Relationship { identifier: Some("r0".to_string()), direction: Direction::Incoming, ..Relationship::default() }; "incoming")]
    #[test_case("<-[ r0 ]-", Relationship { identifier: Some("r0".to_string()), direction: Direction::Incoming, ..Relationship::default() }; "incoming with space")]
    fn relationship_with_identifier(input: &str, expected: Relationship) {
        assert_eq!(input.parse(), Ok(expected));
    }

    #[test_case("-[:BAR]->",   Relationship { rel_type: Some("BAR".to_string()), direction: Direction::Outgoing, ..Relationship::default() }; "outgoing")]
    #[test_case("-[ :BAR ]->", Relationship { rel_type: Some("BAR".to_string()), direction: Direction::Outgoing, ..Relationship::default() }; "outgoing with space")]
    #[test_case("<-[:BAR]-",   Relationship { rel_type: Some("BAR".to_string()), direction: Direction::Incoming, ..Relationship::default() }; "incoming")]
    #[test_case("<-[ :BAR ]-", Relationship { rel_type: Some("BAR".to_string()), direction: Direction::Incoming, ..Relationship::default() }; "incoming with space")]
    fn relationship_with_rel_type(input: &str, expected: Relationship) {
        assert_eq!(input.parse(), Ok(expected));
    }

    #[test_case("-[r0:BAR]->", Relationship { identifier: Some("r0".to_string()), rel_type: Some("BAR".to_string()), direction: Direction::Outgoing, }; "outgoing")]
    #[test_case("-[ r0:BAR ]->", Relationship { identifier: Some("r0".to_string()), rel_type: Some("BAR".to_string()), direction: Direction::Outgoing, }; "outgoing with space")]
    #[test_case("<-[r0:BAR]-", Relationship { identifier: Some("r0".to_string()), rel_type: Some("BAR".to_string()), direction: Direction::Incoming, }; "incoming")]
    #[test_case("<-[ r0:BAR ]-", Relationship { identifier: Some("r0".to_string()), rel_type: Some("BAR".to_string()), direction: Direction::Incoming, }; "incoming with space")]
    fn relationship_full(input: &str, expected: Relationship) {
        assert_eq!(input.parse(), Ok(expected));
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

    #[test_case("(42:A)" ; "numeric identifier")]
    #[test_case("(a)-->(42:A)" ; "numeric identifier one hop")]
    fn path_negative(input: &str) {
        assert!(input.parse::<Path>().is_err())
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

    #[test_case("(a)(b)"; "no comma")]
    #[test_case("(a) (b)"; "one space")]
    #[test_case("(a)  (b)"; "more space")]
    #[test_case("(a),(b)"; "comma")]
    #[test_case("(a), (b)"; "comma and space")]
    #[test_case("(a) ,(b)"; "comma and space in front")]
    #[test_case("(a) , (b)"; "comma and more space")]
    #[test_case("(a)  ,  (b)"; "comma and moore space")]
    #[test_case("(a)  ,  (b),"; "comma and mooore space")]
    #[test_case("(a)\n(b)"; "new line")]
    #[test_case("(a),\n(b)"; "new line and comma on same line")]
    #[test_case("(a)\n,(b)"; "new line and comma on next line")]
    #[test_case("(a)\n\r,(b)\n\r"; "new line at the end")]
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
