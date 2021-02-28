#![allow(dead_code)]

use nom::{
    branch::alt,
    bytes::complete::{tag, take_while1},
    character::complete::{alpha1, alphanumeric1},
    combinator::{map, opt, recognize},
    multi::many0,
    sequence::{pair, preceded, tuple},
    IResult,
};
#[derive(Debug, Default, PartialEq)]
pub struct Node {
    identifier: Option<String>,
    labels: Vec<String>,
}

fn identifier(input: &str) -> IResult<&str, String> {
    map(
        recognize(pair(
            alt((alpha1, tag("_"))),
            many0(alt((alphanumeric1, tag("_")))),
        )),
        |identifier: &str| identifier.to_string(),
    )(input)
}

fn label(input: &str) -> IResult<&str, String> {
    map(
        preceded(
            tag(":"),
            recognize(pair(
                take_while1(|c: char| c.is_uppercase()),
                many0(alphanumeric1),
            )),
        ),
        |label: &str| label.to_string(),
    )(input)
}

fn node(input: &str) -> IResult<&str, Node> {
    map(
        tuple((tag("("), opt(identifier), many0(label), tag(")"))),
        |(_, identifier, labels, _)| Node { identifier, labels },
    )(input)
}

#[cfg(test)]
mod tests {
    use super::*;
    use parameterized::parameterized;

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
        assert!(identifier(input).is_ok())
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
        }
    )]
    fn labels_positive(input: &str) {
        assert!(label(input).is_ok())
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

    #[test]
    fn nodes() {
        assert_eq!(node("()"), Ok(("", Node::default())));
        assert_eq!(
            node("(n0)"),
            Ok((
                "",
                Node {
                    identifier: Some("n0".to_string()),
                    ..Node::default()
                }
            ))
        );

        assert_eq!(
            node("(n0:A:B)"),
            Ok((
                "",
                Node {
                    identifier: Some("n0".to_string()),
                    labels: vec!["A".to_string(), "B".to_string()]
                }
            ))
        );
    }
}
