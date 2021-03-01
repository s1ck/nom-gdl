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

fn node_body(input: &str) -> IResult<&str, (Option<String>, Vec<String>)> {
    tuple((opt(identifier), many0(label)))(input)
}

fn node(input: &str) -> IResult<&str, Node> {
    map(
        delimited(tag("("), node_body, tag(")")),
        |(identifier, labels)| Node { identifier, labels },
    )(input)
}

fn relationship_body(input: &str) -> IResult<&str, (Option<String>, Option<String>)> {
    delimited(tag("["), tuple((opt(identifier), opt(rel_type))), tag("]"))(input)
}

fn relationship(input: &str) -> IResult<&str, Relationship> {
    map(
        alt((
            delimited(tag("-"), opt(relationship_body), tag("->")),
            delimited(tag("<-"), opt(relationship_body), tag("-")),
        )),
        |tuple: Option<(Option<String>, Option<String>)>| match tuple {
            Some(t) => Relationship {
                identifier: t.0,
                rel_type: t.1,
            },
            None => Relationship::default(),
        },
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
    fn node_empty() {
        assert_eq!(node("()"), Ok(("", Node::default())));
    }

    #[test]
    fn node_with_identifier() {
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
    }
    #[test]
    fn node_with_labels() {
        assert_eq!(
            node("(:A)"),
            Ok((
                "",
                Node {
                    labels: vec!["A".to_string()],
                    ..Node::default()
                }
            ))
        );
        assert_eq!(
            node("(:A:B)"),
            Ok((
                "",
                Node {
                    labels: vec!["A".to_string(), "B".to_string()],
                    ..Node::default()
                }
            ))
        );
    }

    #[test]
    fn node_full() {
        assert_eq!(
            node("(n0:A)"),
            Ok((
                "",
                Node {
                    identifier: Some("n0".to_string()),
                    labels: vec!["A".to_string()],
                }
            ))
        );
        assert_eq!(
            node("(n0:A:B)"),
            Ok((
                "",
                Node {
                    identifier: Some("n0".to_string()),
                    labels: vec!["A".to_string(), "B".to_string()],
                }
            ))
        );
    }

    #[test]
    fn relationship_empty() {
        assert_eq!(relationship("-->"), Ok(("", Relationship::default())));
        assert_eq!(relationship("-[]->"), Ok(("", Relationship::default())));
        assert_eq!(relationship("<--"), Ok(("", Relationship::default())));
        assert_eq!(relationship("<-[]-"), Ok(("", Relationship::default())));
    }

    #[test]
    fn relationship_with_identifier() {
        assert_eq!(
            relationship("-[r0]->"),
            Ok((
                "",
                Relationship {
                    identifier: Some("r0".to_string()),
                    ..Relationship::default()
                }
            ))
        );
    }

    #[test]
    fn relationship_with_rel_type() {
        assert_eq!(
            relationship("-[:BAR]->"),
            Ok((
                "",
                Relationship {
                    rel_type: Some("BAR".to_string()),
                    ..Relationship::default()
                }
            ))
        );
    }

    #[test]
    fn relationship_full() {
        assert_eq!(
            relationship("-[r0:BAR]->"),
            Ok((
                "",
                Relationship {
                    identifier: Some("r0".to_string()),
                    rel_type: Some("BAR".to_string()),
                }
            ))
        );
    }
}
