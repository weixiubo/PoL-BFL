use ark_bn254::Fr;
use ark_ff::{BigInteger, PrimeField};
use light_poseidon::{Poseidon, PoseidonHasher};
use num_bigint::{BigInt, BigUint, Sign};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::env;
use std::io::{self, BufRead, Read, Write};

type AppResult<T> = Result<T, String>;

struct CircomPoseidon {
    hashers: HashMap<usize, Poseidon<Fr>>,
    modulus: BigInt,
}

impl CircomPoseidon {
    fn new() -> Self {
        let modulus = BigInt::from_biguint(
            Sign::Plus,
            BigUint::from_bytes_be(&Fr::MODULUS.to_bytes_be()),
        );
        Self {
            hashers: HashMap::new(),
            modulus,
        }
    }

    fn field(&self, value: &str) -> AppResult<Fr> {
        let parsed = BigInt::parse_bytes(value.as_bytes(), 10)
            .ok_or_else(|| format!("invalid decimal field element: {value}"))?;
        let mut reduced = parsed % &self.modulus;
        if reduced.sign() == Sign::Minus {
            reduced += &self.modulus;
        }
        let (_, bytes) = reduced.to_bytes_be();
        Ok(Fr::from_be_bytes_mod_order(&bytes))
    }

    fn hash(&mut self, values: &[String]) -> AppResult<String> {
        if values.is_empty() || values.len() > 12 {
            return Err("Poseidon input count must be in [1, 12]".to_string());
        }
        let fields = values
            .iter()
            .map(|value| self.field(value))
            .collect::<AppResult<Vec<_>>>()?;
        let hasher = match self.hashers.entry(values.len()) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => entry.insert(
                Poseidon::<Fr>::new_circom(values.len())
                    .map_err(|error| format!("cannot initialize Poseidon: {error}"))?,
            ),
        };
        let digest = hasher
            .hash(&fields)
            .map_err(|error| format!("Poseidon hash failed: {error}"))?;
        Ok(BigUint::from_bytes_be(&digest.into_bigint().to_bytes_be()).to_str_radix(10))
    }
}

fn decimal(value: &Value) -> AppResult<String> {
    match value {
        Value::String(text) => {
            BigInt::parse_bytes(text.as_bytes(), 10)
                .ok_or_else(|| format!("invalid decimal value: {text}"))?;
            Ok(text.clone())
        }
        Value::Number(number) => Ok(number.to_string()),
        _ => Err("Poseidon values must be decimal strings or integers".to_string()),
    }
}

fn initial(operation: &Value) -> AppResult<String> {
    operation
        .get("initial")
        .map(decimal)
        .transpose()
        .map(|value| value.unwrap_or_else(|| "0".to_string()))
}

fn array<'a>(value: &'a Value, field: &str) -> AppResult<&'a Vec<Value>> {
    value
        .get(field)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{field} array is required"))
}

fn fold2(
    poseidon: &mut CircomPoseidon,
    values: &[Value],
    mut accumulator: String,
) -> AppResult<String> {
    for value in values {
        accumulator = poseidon.hash(&[accumulator, decimal(value)?])?;
    }
    Ok(accumulator)
}

fn fold3(
    poseidon: &mut CircomPoseidon,
    rows: &[Value],
    mut accumulator: String,
) -> AppResult<String> {
    for row in rows {
        let pair = row
            .as_array()
            .filter(|pair| pair.len() == 2)
            .ok_or_else(|| "fold3 rows require exactly two values".to_string())?;
        accumulator = poseidon.hash(&[accumulator, decimal(&pair[0])?, decimal(&pair[1])?])?;
    }
    Ok(accumulator)
}

fn fold_pair_chunks(
    poseidon: &mut CircomPoseidon,
    operation: &Value,
    rows: &[Value],
    mut accumulator: String,
) -> AppResult<String> {
    let chunk_size = operation
        .get("pairs_per_chunk")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .filter(|value| *value > 0)
        .ok_or_else(|| "pairs_per_chunk must be positive".to_string())?;
    if rows.len() % chunk_size != 0 {
        return Err("pair rows must divide exactly into commitment chunks".to_string());
    }
    for chunk in rows.chunks(chunk_size) {
        let mut inputs = Vec::with_capacity(1 + 2 * chunk_size);
        inputs.push(accumulator);
        for row in chunk {
            let pair = row
                .as_array()
                .filter(|pair| pair.len() == 2)
                .ok_or_else(|| "pair row requires exactly two values".to_string())?;
            inputs.push(decimal(&pair[0])?);
            inputs.push(decimal(&pair[1])?);
        }
        accumulator = poseidon.hash(&inputs)?;
    }
    Ok(accumulator)
}

fn handle(poseidon: &mut CircomPoseidon, request: &Value) -> AppResult<Value> {
    let operations = array(request, "operations")?;
    let mut results = Vec::with_capacity(operations.len());
    for operation in operations {
        let kind = operation
            .get("kind")
            .and_then(Value::as_str)
            .ok_or_else(|| "operation kind is required".to_string())?;
        let accumulator = initial(operation)?;
        let result = match kind {
            "fold2" => fold2(poseidon, array(operation, "values")?, accumulator)?,
            "fold3" => fold3(poseidon, array(operation, "rows")?, accumulator)?,
            "fold_pair_chunks" => {
                fold_pair_chunks(poseidon, operation, array(operation, "rows")?, accumulator)?
            }
            _ => return Err(format!("unsupported Poseidon operation: {kind}")),
        };
        results.push(Value::String(result));
    }
    Ok(json!({"results": results}))
}

fn process(poseidon: &mut CircomPoseidon, input: &str) -> AppResult<Value> {
    let request: Value = serde_json::from_str(input)
        .map_err(|error| format!("request is not valid JSON: {error}"))?;
    handle(poseidon, &request)
}

fn self_test() -> AppResult<String> {
    let mut poseidon = CircomPoseidon::new();
    let observed = poseidon.hash(&["1".to_string(), "2".to_string()])?;
    let expected = BigUint::parse_bytes(
        b"115cc0f5e7d690413df64c6b9662e9cf2a3617f2743245519e19607a4417189a",
        16,
    )
    .ok_or_else(|| "invalid embedded acceptance vector".to_string())?
    .to_str_radix(10);
    if observed != expected {
        return Err(format!(
            "Circom acceptance vector mismatch: expected {expected}, observed {observed}"
        ));
    }
    Ok(observed)
}

fn main() {
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    if arguments.iter().any(|argument| argument == "--version") {
        println!("polbfl-poseidon-native {}", env!("CARGO_PKG_VERSION"));
        return;
    }
    if arguments.iter().any(|argument| argument == "--self-test") {
        match self_test() {
            Ok(digest) => println!("{digest}"),
            Err(error) => {
                eprintln!("{error}");
                std::process::exit(1);
            }
        }
        return;
    }

    let stream_mode = arguments.iter().any(|argument| argument == "--stream");
    let mut poseidon = CircomPoseidon::new();
    if stream_mode {
        let stdin = io::stdin();
        let mut stdout = io::BufWriter::new(io::stdout().lock());
        for line in stdin.lock().lines() {
            let response = match line {
                Ok(line) => {
                    process(&mut poseidon, &line).unwrap_or_else(|error| json!({"error": error}))
                }
                Err(error) => json!({"error": format!("cannot read request: {error}")}),
            };
            if writeln!(stdout, "{response}").is_err() || stdout.flush().is_err() {
                std::process::exit(1);
            }
        }
        return;
    }

    let mut input = String::new();
    if let Err(error) = io::stdin().read_to_string(&mut input) {
        eprintln!("cannot read request: {error}");
        std::process::exit(1);
    }
    match process(&mut poseidon, &input) {
        Ok(response) => println!("{response}"),
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    }
}
