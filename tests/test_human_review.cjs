// Synthetic fixtures only: never write these as actual reviewer judgments.
const assert = require('node:assert/strict');
const {complete, validateBatch} = require('../assets/human_review.js');
const blank = {task:'retrieval',batch_id:'r',query_id:'q',product_id:'p',grade:null,reviewer:'',note:'',review_origin:null};
assert.equal(validateBatch([blank],[blank]).completed, 0);
const human = {...blank, grade:0, reviewer:'synthetic-test', review_origin:'human'};
assert.equal(complete(human), true);
for (const changed of [{grade:false},{grade:'0'},{reviewer:''},{review_origin:'model_assisted'},{batch_id:'other'},{product_id:'other'},{note:42}]) {
  assert.throws(()=>validateBatch([blank],[{...human,...changed}]));
}
assert.throws(()=>validateBatch([blank],[]));
assert.throws(()=>validateBatch([blank,{...blank,product_id:'other'}],[blank,blank]));
const copy = {task:'copy',batch_id:'c',id:'p',status:'blocked_by_rules',reviewer:'',note:'',review_origin:null,relevance_pass:null,fluency_pass:null,factual_error:null};
const failed = {...copy,reviewer:'synthetic-test',review_origin:'human',relevance_pass:false,fluency_pass:true,factual_error:true};
assert.equal(complete(failed),false);
assert.equal(complete({...failed,note:'Synthetic source contradiction.'}),true);
assert.throws(()=>validateBatch([copy],[{...failed,relevance_pass:true}]));
const blocked = {...copy,status:'blocked_source_conflict'};
assert.throws(()=>validateBatch([blocked],[{...blocked,reviewer:'synthetic-test',review_origin:'human',fluency_pass:false}]));
assert.equal(complete({...blocked,reviewer:'synthetic-test',review_origin:'human',note:'Confirmed synthetic block.'}),true);
console.log('Offline review validation: 16 assertions passed.');
