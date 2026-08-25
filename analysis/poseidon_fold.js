// Poseidon fold hash (arity-2) using circomlibjs
// Usage: node scripts/poseidon_fold.js '[1,2,3]'
// Prints a decimal string representing the field element

const cir = require('circomlibjs');

async function main() {
  const arg = process.argv[2];
  if (!arg) {
    console.error('Usage: node scripts/poseidon_fold.js "[1,2,3]"');
    process.exit(1);
  }
  const arr = JSON.parse(arg);
  const poseidon = await cir.buildPoseidon();

  let acc = 0n;
  for (const x of arr) {
    acc = poseidon([acc, BigInt(x)]);
  }
  const out = poseidon.F.toString(acc);
  process.stdout.write(out);
}

main().catch((error) => { console.error(error); process.exit(1); });

