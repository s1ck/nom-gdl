use std::{collections::HashMap, fmt::Display, str::FromStr};

use nom::{
    branch::alt,
    bytes::complete::{escaped, tag, take_while, take_while1},
    character::complete::{alpha1, alphanumeric1, digit0, digit1, none_of},
    combinator::{all_consuming, cut, map, opt, recognize},
    error::Error,
    multi::{many0, many1},
    sequence::{delimited, pair, preceded, terminated, tuple},
    Finish, IResult,
};

#[derive(Debug, Default, PartialEq)]
pub(crate) struct Node {
    pub(crate) variable: Option<String>,
    pub(crate) labels: Vec<String>,
    pub(crate) properties: HashMap<String, CypherValue>,
}

impl Node {
    pub fn new(
        variable: Option<String>,
        labels: Vec<String>,
        properties: HashMap<String, CypherValue>,
    ) -> Self {
        Self {
            variable,
            labels,
            properties,
        }
    }
}

impl
    From<(
        Option<String>,
        Vec<String>,
        Option<HashMap<String, CypherValue>>,
    )> for Node
{
    fn from(
        (variable, labels, properties): (
            Option<String>,
            Vec<String>,
            Option<HashMap<String, CypherValue>>,
        ),
    ) -> Self {
        Node {
            variable,
            labels,
            properties: properties.unwrap_or_default(),
        }
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

#[derive(Debug, Default, PartialEq)]
pub(crate) struct Relationship {
    pub(crate) variable: Option<String>,
    pub(crate) rel_type: Option<String>,
    pub(crate) direction: Direction,
    pub(crate) properties: HashMap<String, CypherValue>,
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
    fn outgoing(
        variable: Option<String>,
        rel_type: Option<String>,
        properties: HashMap<String, CypherValue>,
    ) -> Self {
        Relationship {
            variable,
            rel_type,
            direction: Direction::Outgoing,
            properties,
        }
    }

    fn incoming(
        variable: Option<String>,
        rel_type: Option<String>,
        properties: HashMap<String, CypherValue>,
    ) -> Self {
        Relationship {
            variable,
            rel_type,
            direction: Direction::Incoming,
            properties,
        }
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

#[derive(Debug, PartialEq, Default)]
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

#[derive(Debug, PartialEq, Default)]
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
    String(String),
    Boolean(bool),
}

impl From<f64> for CypherValue {
    fn from(value: f64) -> Self {
        CypherValue::Float(value)
    }
}

impl From<i64> for CypherValue {
    fn from(value: i64) -> Self {
        CypherValue::Integer(value)
    }
}

impl From<String> for CypherValue {
    fn from(value: String) -> Self {
        CypherValue::String(value)
    }
}

impl From<&str> for CypherValue {
    fn from(value: &str) -> Self {
        CypherValue::String(value.into())
    }
}

impl From<bool> for CypherValue {
    fn from(value: bool) -> Self {
        CypherValue::Boolean(value)
    }
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

impl Display for CypherValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CypherValue::Float(float)       => write!(f, "{}", float),
            CypherValue::Integer(integer)   => write!(f, "{}", integer),
            CypherValue::String(string)     => f.pad(string),
            CypherValue::Boolean(boolean)   => write!(f, "{}", boolean),
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
        CypherValue::from(num)
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
            CypherValue::from(num)
        },
    )(input)
}

fn single_quoted_string(input: &str) -> IResult<&str, &str> {
    let escaped = escaped(none_of("\\\'"), '\\', tag("'"));
    let escaped_or_empty = alt((escaped, tag("")));
    delimited(tag("'"), escaped_or_empty, tag("'"))(input)
}

fn double_quoted_string(input: &str) -> IResult<&str, &str> {
    let escaped = escaped(none_of("\\\""), '\\', tag("\""));
    let escaped_or_empty = alt((escaped, tag("")));
    delimited(tag("\""), escaped_or_empty, tag("\""))(input)
}

fn string_literal(input: &str) -> IResult<&str, CypherValue> {
    map(
        alt((single_quoted_string, double_quoted_string)),
        |literal: &str| CypherValue::from(literal),
    )(input)
}

fn boolean_literal(input: &str) -> IResult<&str, CypherValue> {
  alt((
      map(tag("false"), |_| CypherValue::Boolean(false)),
      map(tag("true"), |_| CypherValue::Boolean(true))
  ))(input)
}

fn cypher_value(input: &str) -> IResult<&str, CypherValue> {
    preceded(sp, alt((float_literal, integer_literal, string_literal, boolean_literal)))(input)
}

fn variable(input: &str) -> IResult<&str, String> {
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
        variable,
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

fn node_body(
    input: &str,
) -> IResult<
    &str,
    (
        Option<String>,
        Vec<String>,
        Option<HashMap<String, CypherValue>>,
    ),
> {
    delimited(
        sp,
        tuple((opt(variable), many0(label), opt(properties))),
        sp,
    )(input)
}

pub(crate) fn node(input: &str) -> IResult<&str, Node> {
    map(delimited(tag("("), node_body, tag(")")), Node::from)(input)
}

fn relationship_body(
    input: &str,
) -> IResult<
    &str,
    (
        Option<String>,
        Option<String>,
        Option<HashMap<String, CypherValue>>,
    ),
> {
    delimited(
        tag("["),
        delimited(
            sp,
            tuple((opt(variable), opt(rel_type), opt(properties))),
            sp,
        ),
        tag("]"),
    )(input)
}

pub(crate) fn relationship(input: &str) -> IResult<&str, Relationship> {
    alt((
        map(
            delimited(tag("-"), opt(relationship_body), tag("->")),
            |relationship| match relationship {
                Some((variable, rel_type, properties)) => {
                    Relationship::outgoing(variable, rel_type, properties.unwrap_or_default())
                }
                None => Relationship::outgoing(None, None, HashMap::default()),
            },
        ),
        map(
            delimited(tag("<-"), opt(relationship_body), tag("-")),
            |relationship| match relationship {
                Some((variable, rel_type, properties)) => {
                    Relationship::incoming(variable, rel_type, properties.unwrap_or_default())
                }
                None => Relationship::incoming(None, None, HashMap::default()),
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

    impl Node {
        fn with_variable(variable: impl Into<String>) -> Self {
            Node {
                variable: Some(variable.into()),
                ..Node::default()
            }
        }

        fn with_labels<I, T>(labels: I) -> Self
        where
            I: IntoIterator<Item = T>,
            T: Into<String>,
        {
            Node {
                labels: labels.into_iter().map(Into::into).collect(),
                ..Node::default()
            }
        }

        fn with_variable_and_labels<I, T>(variable: impl Into<String>, labels: I) -> Self
        where
            I: IntoIterator<Item = T>,
            T: Into<String>,
        {
            Node {
                variable: Some(variable.into()),
                labels: labels.into_iter().map(Into::into).collect(),
                ..Node::default()
            }
        }

        fn from<I, T>(
            variable: impl Into<String>,
            labels: I,
            properties: Vec<(T, CypherValue)>,
        ) -> Self
        where
            I: IntoIterator<Item = T>,
            T: Into<String>,
        {
            Node {
                variable: Some(variable.into()),
                labels: labels.into_iter().map(Into::into).collect(),
                properties: properties
                    .into_iter()
                    .map(|(k, v)| (Into::into(k), v))
                    .collect::<HashMap<_, _>>(),
            }
        }
    }

    impl Relationship {
        fn outgoing_with_variable(variable: impl Into<String>) -> Self {
            Self::outgoing(Some(variable.into()), None, HashMap::default())
        }

        fn outgoing_with_variable_and_rel_type(
            variable: impl Into<String>,
            rel_type: impl Into<String>,
        ) -> Self {
            Self::outgoing(
                Some(variable.into()),
                Some(rel_type.into()),
                HashMap::default(),
            )
        }

        fn incoming_with_variable(variable: impl Into<String>) -> Self {
            Self::incoming(Some(variable.into()), None, HashMap::default())
        }

        fn incoming_with_variable_and_rel_type(
            variable: impl Into<String>,
            rel_type: impl Into<String>,
        ) -> Self {
            Self::incoming(
                Some(variable.into()),
                Some(rel_type.into()),
                HashMap::default(),
            )
        }

        fn from<I, T>(
            variable: impl Into<String>,
            rel_type: T,
            direction: Direction,
            properties: Vec<(T, CypherValue)>,
        ) -> Self
        where
            T: Into<String>,
        {
            Relationship {
                variable: Some(variable.into()),
                rel_type: Some(rel_type.into()),
                direction,
                properties: properties
                    .into_iter()
                    .map(|(k, v)| (Into::into(k), v))
                    .collect::<HashMap<_, _>>(),
            }
        }
    }

    #[test_case("0",              CypherValue::from(0)   ; "int: zero")]
    #[test_case("-0",             CypherValue::from(0)   ; "int: signed zero")]
    #[test_case("42",             CypherValue::from(42)  ; "int: positive")]
    #[test_case("-42",            CypherValue::from(-42) ; "int: negative")]
    #[test_case("0.0",            CypherValue::from(0.0)   ; "float: zero v1")]
    #[test_case("0.",             CypherValue::from(0.0)   ; "float: zero v2")]
    #[test_case(".0",             CypherValue::from(0.0)   ; "float: zero v3")]
    #[test_case("-0.0",           CypherValue::from(0.0)   ; "float: signed zero")]
    #[test_case("-.0",            CypherValue::from(0.0)   ; "float: signed zero v2")]
    #[test_case("13.37",          CypherValue::from(13.37) ; "float: positive")]
    #[test_case("-42.2",          CypherValue::from(-42.2) ; "float: negative")]
    #[test_case("'foobar'",       CypherValue::from("foobar")       ; "sq string: alpha")]
    #[test_case("'1234'",         CypherValue::from("1234")         ; "sq string: numeric")]
    #[test_case("'    '",         CypherValue::from("    ")         ; "sq string: whitespacec")]
    #[test_case("''",             CypherValue::from("")             ; "sq string: empty")]
    #[test_case(r#"'foobar\'s'"#, CypherValue::from(r#"foobar\'s"#) ; "sq string: escaped")]
    #[test_case("\"foobar\"",     CypherValue::from("foobar")       ; "dq string: alpha")]
    #[test_case("\"1234\"",       CypherValue::from("1234")         ; "dq string: numeric")]
    #[test_case("\"    \"",       CypherValue::from("    ")         ; "dq string: whitespacec")]
    #[test_case("\"\"",           CypherValue::from("")             ; "dq string: empty")]
    #[test_case(r#""foobar\"s""#, CypherValue::from(r#"foobar\"s"#) ; "dq string: escaped")]
    #[test_case("true",           CypherValue::from(true)  ; "bool: true")]
    #[test_case("false",          CypherValue::from(false) ; "bool: false")]
    fn cypher_value(input: &str, expected: CypherValue) {
        assert_eq!(input.parse(), Ok(expected))
    }

    #[test]
    fn cypher_value_from() {
        assert_eq!(CypherValue::from(42), CypherValue::from(42));
        assert_eq!(CypherValue::from(13.37), CypherValue::from(13.37));
        assert_eq!(CypherValue::from("foobar"), CypherValue::from("foobar"));
        assert_eq!(CypherValue::from(true), CypherValue::from(true));
    }

    #[test]
    fn cypher_value_display() {
        assert_eq!("42", format!("{}", CypherValue::from(42)));
        assert_eq!("13.37", format!("{}", CypherValue::from(13.37)));
        assert_eq!("foobar", format!("{}", CypherValue::from("foobar")));
        assert_eq!("00foobar", format!("{:0>8}", CypherValue::from("foobar")));
        assert_eq!("true", format!("{}", CypherValue::from(true)));
    }

    #[test_case("key:42",         ("key".to_string(), CypherValue::from(42)))]
    #[test_case("key: 1337",      ("key".to_string(), CypherValue::from(1337)))]
    #[test_case("key2: 1337",     ("key2".to_string(), CypherValue::from(1337)))]
    #[test_case("key2: 'foobar'", ("key2".to_string(), CypherValue::from("foobar")))]
    #[test_case("key3: true",     ("key3".to_string(), CypherValue::from(true)))]
    fn key_value_pair_test(input: &str, expected: (String, CypherValue)) {
        assert_eq!(key_value_pair(input).unwrap().1, expected)
    }

    #[test_case("{key1: 42}",               vec![("key1".to_string(), CypherValue::from(42))])]
    #[test_case("{key1: 13.37 }",           vec![("key1".to_string(), CypherValue::from(13.37))])]
    #[test_case("{ key1: 42, key2: 1337 }", vec![("key1".to_string(), CypherValue::from(42)), ("key2".to_string(), CypherValue::from(1337))])]
    #[test_case("{ key1: 42, key2: 1337, key3: 'foobar' }", vec![("key1".to_string(), CypherValue::from(42)), ("key2".to_string(), CypherValue::from(1337)), ("key3".to_string(), CypherValue::from("foobar"))])]
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
    fn variable_positive(input: &str) {
        let result = variable(input);
        assert!(result.is_ok());
        let result = result.unwrap().1;
        assert_eq!(result, input)
    }

    #[test_case("1234"; "numerical")]
    #[test_case("+foo"; "special char")]
    #[test_case("."; "another special char")]
    fn variable_negative(input: &str) {
        assert!(variable(input).is_err())
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

    #[test_case("()",                   Node::default(); "empty node")]
    #[test_case("( )",                  Node::default(); "empty node with space")]
    #[test_case("(  )",                 Node::default(); "empty node with many spaces")]
    #[test_case("(n0)",                 Node::with_variable("n0"); "n0")]
    #[test_case("( n0 )",               Node::with_variable("n0"); "n0 with space")]
    #[test_case("(:A)",                 Node::with_labels(vec!["A"]))]
    #[test_case("(:A:B)",               Node::with_labels(vec!["A", "B"]) )]
    #[test_case("( :A:B )",             Node::with_labels(vec!["A", "B"]); ":A:B with space" )]
    #[test_case("(n0:A)",               Node::with_variable_and_labels("n0", vec!["A"]))]
    #[test_case("(n0:A:B)",             Node::with_variable_and_labels("n0", vec!["A", "B"]))]
    #[test_case("( n0:A:B )",           Node::with_variable_and_labels("n0", vec!["A", "B"]); "n0:A:B with space")]
    #[test_case("(n0 { foo: 42 })",     Node::from("n0", vec![], vec![("foo", CypherValue::from(42))]))]
    #[test_case("(n0:A:B { foo: 42 })", Node::from("n0", vec!["A", "B"], vec![("foo", CypherValue::from(42))]))]
    fn node_test(input: &str, expected: Node) {
        assert_eq!(input.parse(), Ok(expected));
    }

    #[test_case("(42:A)" ; "numeric variable")]
    #[test_case("("      ; "no closing")]
    #[test_case(")"      ; "no opening")]
    fn node_negative(input: &str) {
        assert!(input.parse::<Node>().is_err())
    }

    #[test_case("-->",                     Relationship { direction: Direction::Outgoing, ..Relationship::default()})]
    #[test_case("-[]->",                   Relationship { direction: Direction::Outgoing, ..Relationship::default()}; "outgoing: with body")]
    #[test_case("-[ ]->",                  Relationship { direction: Direction::Outgoing, ..Relationship::default()}; "outgoing: body with space")]
    #[test_case("<--",                     Relationship { direction: Direction::Incoming, ..Relationship::default()}; "incoming: no body")]
    #[test_case("<-[]-",                   Relationship { direction: Direction::Incoming, ..Relationship::default()}; "incoming: with body")]
    #[test_case("<-[ ]-",                  Relationship { direction: Direction::Incoming, ..Relationship::default()}; "incoming: body with space")]
    #[test_case("-[r0]->",                 Relationship { variable: Some("r0".to_string()), direction: Direction::Outgoing, ..Relationship::default() }; "r0 outgoing")]
    #[test_case("-[ r0 ]->",               Relationship { variable: Some("r0".to_string()), direction: Direction::Outgoing, ..Relationship::default() }; "r0 outgoing with space")]
    #[test_case("<-[r0]-",                 Relationship { variable: Some("r0".to_string()), direction: Direction::Incoming, ..Relationship::default() }; "r0 incoming")]
    #[test_case("<-[ r0 ]-",               Relationship { variable: Some("r0".to_string()), direction: Direction::Incoming, ..Relationship::default() }; "r0 incoming with space")]
    #[test_case("-[:BAR]->",               Relationship { rel_type: Some("BAR".to_string()), direction: Direction::Outgoing, ..Relationship::default() }; "BAR outgoing")]
    #[test_case("-[ :BAR ]->",             Relationship { rel_type: Some("BAR".to_string()), direction: Direction::Outgoing, ..Relationship::default() }; "BAR outgoing with space")]
    #[test_case("<-[:BAR]-",               Relationship { rel_type: Some("BAR".to_string()), direction: Direction::Incoming, ..Relationship::default() }; "BAR incoming")]
    #[test_case("<-[ :BAR ]-",             Relationship { rel_type: Some("BAR".to_string()), direction: Direction::Incoming, ..Relationship::default() }; "BAR incoming with space")]
    #[test_case("-[r0:BAR]->",             Relationship { variable: Some("r0".to_string()), rel_type: Some("BAR".to_string()), direction: Direction::Outgoing, ..Relationship::default() }; "r0:Bar outgoing")]
    #[test_case("-[ r0:BAR ]->",           Relationship { variable: Some("r0".to_string()), rel_type: Some("BAR".to_string()), direction: Direction::Outgoing, ..Relationship::default()};  "r0:Bar outgoing with space")]
    #[test_case("<-[r0:BAR]-",             Relationship { variable: Some("r0".to_string()), rel_type: Some("BAR".to_string()), direction: Direction::Incoming, ..Relationship::default() }; "r0:Bar incoming")]
    #[test_case("<-[ r0:BAR ]-",           Relationship { variable: Some("r0".to_string()), rel_type: Some("BAR".to_string()), direction: Direction::Incoming, ..Relationship::default() }; "r0:Bar incoming with space")]
    #[test_case("<-[{ foo: 42 }]-",        Relationship { variable: None, rel_type: None, direction: Direction::Incoming, properties: std::iter::once(("foo".to_string(), CypherValue::from(42))).into_iter().collect::<HashMap<_,_>>() }; "with properties")]
    #[test_case("<-[r0 { foo: 42 }]-",     Relationship { variable: Some("r0".to_string()), rel_type: None, direction: Direction::Incoming, properties: std::iter::once(("foo".to_string(), CypherValue::from(42))).into_iter().collect::<HashMap<_,_>>() }; "r0 with properties")]
    #[test_case("<-[:BAR { foo: 42 }]-",   Relationship { variable: None, rel_type: Some("BAR".to_string()), direction: Direction::Incoming, properties: std::iter::once(("foo".to_string(), CypherValue::from(42))).into_iter().collect::<HashMap<_,_>>() }; "Bar with properties")]
    #[test_case("<-[r0:BAR { foo: 42 }]-", Relationship { variable: Some("r0".to_string()), rel_type: Some("BAR".to_string()), direction: Direction::Incoming, properties: std::iter::once(("foo".to_string(), CypherValue::from(42))).into_iter().collect::<HashMap<_,_>>() }; "r0:Bar with properties")]
    fn relationship_test(input: &str, expected: Relationship) {
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

    #[test]
    fn path_node_only() {
        assert_eq!(
            "(a)".parse(),
            Ok(Path {
                start: Node::with_variable("a"),
                elements: vec![]
            })
        );
    }

    #[test]
    fn path_one_hop_path() {
        assert_eq!(
            "(a)-->(b)".parse(),
            Ok(Path {
                start: Node::with_variable("a"),
                elements: vec![(
                    Relationship::outgoing(None, None, HashMap::default()),
                    Node::with_variable("b")
                )]
            })
        );
    }

    #[test]
    fn path_two_hop_path() {
        assert_eq!(
            "(a)-->(b)<--(c)".parse(),
            Ok(Path {
                start: Node::with_variable("a"),
                elements: vec![
                    (
                        Relationship::outgoing(None, None, HashMap::default()),
                        Node::with_variable("b")
                    ),
                    (
                        Relationship::incoming(None, None, HashMap::default()),
                        Node::with_variable("c")
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
                start: Node::with_variable_and_labels("a", vec!["A"]),
                elements: vec![
                    (
                        Relationship::incoming(None, Some("R".to_string()), HashMap::default()),
                        Node::new(None, vec!["B".to_string()], HashMap::default())
                    ),
                    (
                        Relationship::outgoing_with_variable("rel"),
                        Node::with_variable("c")
                    ),
                    (
                        Relationship::outgoing(None, None, HashMap::default()),
                        Node::with_variable_and_labels("d", vec!["D1", "D2"])
                    ),
                    (
                        Relationship::incoming(None, None, HashMap::default()),
                        Node::new(
                            None,
                            vec!["E1".to_string(), "E2".to_string()],
                            HashMap::default()
                        )
                    ),
                    (
                        Relationship::outgoing_with_variable_and_rel_type("r", "REL"),
                        Node::default()
                    ),
                ]
            })
        );
    }

    #[test_case("(42:A)" ; "numeric variable")]
    #[test_case("(a)-->(42:A)" ; "numeric variable one hop")]
    fn path_negative(input: &str) {
        assert!(input.parse::<Path>().is_err())
    }

    #[test]
    fn graph_one_paths() {
        pretty_assert_eq!(
            "(a)-->(b)".parse(),
            Ok(Graph::new(vec![Path {
                start: Node::with_variable("a"),
                elements: vec![(
                    Relationship::outgoing(None, None, HashMap::default()),
                    Node::with_variable("b")
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
                    start: Node::with_variable("a"),
                    elements: vec![]
                },
                Path {
                    start: Node::with_variable("b"),
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
                start: Node::with_variable("a"),
                elements: vec![]
            }]))
        );
    }
}
