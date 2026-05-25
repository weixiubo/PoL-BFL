// Poseidon fold hash (arity-2) using circomlibjs
// Usage: node scripts/poseidon_fold.js '[1,2,3]'
// Prints a decimal string representing the field element

const cir = require('circomlibjs');

function main() {
  const arg = process.argv[2];
  if (!arg) {
    console.error('Usage: node scripts/poseidon_fold.js "[1,2,3]"');
    process.exit(1);
  }
  const arr = JSON.parse(arg);
  const poseidon = cir.poseidon; // circomlibjs 0.0.8 legacy Poseidon

  let acc = 0n; // BigInt accumulator in Fr
  for (const x of arr) {
    acc = poseidon([acc, BigInt(x)]);
  }
  const out = acc.toString();
  process.stdout.write(out);
}

try { main(); } catch (e) { console.error(e); process.exit(1); }

