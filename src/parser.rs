#![allow(dead_code)]

use nom::{
    branch::alt,
    bytes::complete::{tag, take_while, take_while1},
    character::complete::{alpha1, alphanumeric1},
    combinator::{map, opt, recognize},
    multi::many0,
    sequence::{delimited, pair, preceded, tuple},
    IResult,
};
#[derive(Debug, Default, PartialEq)]
pub struct Node {
    identifier: Option<String>,
    labels: Vec<String>,
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
        |identifier: &str| identifier.to_string(),
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
        |label: &str| label.to_string(),
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
        |rel_type: &str| rel_type.to_string(),
    )(input)
}

fn node(input: &str) -> IResult<&str, Node> {
    map(
        delimited(tag("("), tuple((opt(identifier), many0(label))), tag(")")),
        |(identifier, labels)| Node { identifier, labels },
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
