use core::slice;
use std::fs::File;
use std::io::Read;
use std::vec;

use crate::config::LlamaConfigJson;
use crate::kvcache::KVCache;
use crate::operators::{self as OP, copy_mat, matadd, matmul_transb, rms_norm, silu};
use crate::params::LLamaParams;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
use std::path::Path;
pub struct Llama<T> {
    vocab: usize,           // vocab size
    n_layers: usize,        // number of layers
    n_q_h: usize,           // number of heads for q
    n_kv_h: usize,          // number of heads for k and v
    d: usize,               // dimension of hidden states
    dqkv: usize,            // length of a single q, k, or v vector
    di: usize,              // dimension of intermediate states
    eps: f32,               // epsilon for RMS normalization
    rope_theta: f32,        // rope theta for rope initialization
    max_seq_len: usize,     // maximum sequence length
    params: LLamaParams<T>, // trained weights of this model
    bos_token_id: u32,      // start token id
    eos_token_id: u32,      // end token id
}

impl Llama<f32> {
    pub fn from_safetensors(model_dir: impl AsRef<Path>) -> Self {
        let config = File::open(model_dir.as_ref().join("config.json")).unwrap();
        let config: LlamaConfigJson = serde_json::from_reader(config).unwrap();
        let model_file = std::fs::read(model_dir.as_ref().join("model.safetensors")).unwrap();
        let safetensor = SafeTensors::deserialize(&model_file).unwrap();
        let params = LLamaParams::from_safetensors(&safetensor, &config);

        Self {
            vocab: config.vocab_size,
            n_layers: config.num_hidden_layers,
            n_q_h: config.num_attention_heads,
            n_kv_h: config.num_key_value_heads,
            d: config.hidden_size,
            dqkv: config.hidden_size / config.num_attention_heads,
            di: config.intermediate_size,
            eps: config.rms_norm_eps,
            rope_theta: config.rope_theta,
            max_seq_len: config.max_position_embeddings,
            params: params,
            bos_token_id: config.bos_token_id,
            eos_token_id: config.eos_token_id,
        }
    }

    pub fn new_cache(&self) -> KVCache<f32> {
        KVCache::new(self.n_layers, self.max_seq_len, self.n_kv_h * self.dqkv, 0)
    }

    pub fn forward(&self, input: &Tensor<u32>, cache: &mut KVCache<f32>) -> Tensor<f32> {
        let seq_len = input.size();
        let past_seq_len = cache.len();
        cache.increment(seq_len);
        let total_seq_len = past_seq_len + seq_len;
        let n_groups = self.n_q_h / self.n_kv_h;

        // Some pre-allocated buffers that will be reused
        let mut residual = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut q_buf = Tensor::<f32>::default(&vec![seq_len, self.n_q_h * self.dqkv]);
        let mut att_scores =
            Tensor::<f32>::default(&vec![self.n_kv_h, n_groups, seq_len, total_seq_len]);
        let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);
        let mut up_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);

        // Computation Starts Here
        // Embedding lookup
        OP::gather(&mut residual, input, &self.params.embedding_table);

        for layer in 0..self.n_layers {
            OP::rms_norm(
                &mut hidden_states,
                &residual,
                &self.params.rms_att_w[layer],
                self.eps,
            );

            let q = (&mut q_buf).reshape(&vec![seq_len, self.n_q_h * self.dqkv]); // (seq, n_h * dqkv)
            let k = &mut cache.k_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            let v = &mut cache.v_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            OP::matmul_transb(q, 0., &hidden_states, &self.params.wq[layer], 1.0);
            OP::matmul_transb(k, 0., &hidden_states, &self.params.wk[layer], 1.0);
            OP::matmul_transb(v, 0., &hidden_states, &self.params.wv[layer], 1.0);
            OP::rope(
                q.reshape(&vec![seq_len, self.n_q_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );
            OP::rope(
                k.reshape(&vec![seq_len, self.n_kv_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );

            let full_k = &mut cache.k_cache(layer, 0); // (total_seq, n_kv_h * dqkv)
            let full_v = &mut cache.v_cache(layer, 0); // (total_seq, n_kv_h * dqkv)

            // self_attention(&mut hidden_states, &mut att_scores, q, k, v, n_kv_h, n_groups, seq_len, total_seq_len, dqkv);
            self_attention(&mut hidden_states, &mut att_scores, q, k, v, self.n_kv_h, n_groups, seq_len, total_seq_len, self.dqkv);
            
            let mut hidden_states_clone = Tensor::<f32>::default(&vec![seq_len, self.n_kv_h * n_groups * self.dqkv]);
            OP::copy_mat(&mut hidden_states_clone, &hidden_states);
            OP::matmul_transb(&mut hidden_states, 0., &hidden_states_clone, &self.params.wo[layer], 1.0);

            // 添加残差
            residual = OP::matadd(&hidden_states, &residual);


            // todo!("down_proj matmul and add residual");

            // todo!("mlp(...)");
            mlp(&mut residual, &mut hidden_states, &mut gate_buf, &mut up_buf, 
                &self.params.w_up[layer], &self.params.w_down[layer], &self.params.w_gate[layer], 
                &self.params.rms_out_w, self.eps
            );
            
        }

        // No matter what seq_len, the output is always a 1D vector of length vocab,
        // which contains the probabilities for the next token.
        let mut logits = Tensor::<f32>::default(&vec![1, self.vocab]);
        let mut hidden_states = hidden_states.slice((seq_len - 1) * self.d, &vec![1, self.d]);
        let residual = residual.slice((seq_len - 1) * self.d, &vec![self.d]);

        OP::rms_norm(
            &mut hidden_states,
            &residual,
            &self.params.rms_out_w,
            self.eps,
        );

        OP::matmul_transb(&mut logits, 0., &hidden_states, &self.params.lm_head, 1.0);

        logits
    }

    pub fn generate(
        &self,
        token_ids: &[u32],
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
    ) -> Vec<u32>{
        let mut result = Vec::<u32>::new();
        
        todo!("实现文本生成");
        
        result
    }
}

/// 待测试的函数
fn self_attention(
    hidden_states: &mut Tensor<f32>, // (seq, n_kv_h * n_groups * dqkv)
    att_scores: &mut Tensor<f32>,    // (n_kv_h, n_groups, seq, total_seq)
    q: &Tensor<f32>,                 // (seq, n_kv_h * n_groups * dqkv)
    k: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
    v: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
    n_kv_h: usize,
    n_groups: usize,
    seq_len: usize,
    total_seq_len: usize,
    dqkv: usize,
) {
    // Step 1: 计算 QK^T / sqrt(dim)
    let dim = dqkv as f32;
    let sqrt_dim = dim.sqrt();

    for g in 0..n_groups {
        for h in 0..n_kv_h {
            // 取出Q的当前头
            let q_slice = q.slice(g * n_kv_h + h, &vec![seq_len, dqkv]);
            // 取出K的当前头
            let k_slice = k.slice(h, &vec![total_seq_len, dqkv]);

            // 计算 QK^T / sqrt(dim)
            let mut score = Tensor::<f32>::default(&vec![seq_len, total_seq_len]);
            OP::matmul_transb(&mut score, 0., &q_slice, &k_slice, 1. / sqrt_dim);
            
            // 保存得分
            for i in 0..seq_len {
                for j in 0..total_seq_len {
                    // 将 score 的值复制到 att_scores 对应位置
                    unsafe {
                        att_scores.data_mut()[h * n_groups * seq_len * total_seq_len
                                          + g * seq_len * total_seq_len
                                          + i * total_seq_len
                                          + j] = score.data()[i * total_seq_len + j];
                    }
                    
                }
            }
            // warning 不确定点
            // Step 2: 计算 softmax 
            OP::masked_softmax(att_scores);

            // Step 3: 计算注意力输出 attn @ V
            // x = attn @ V
            OP::matmul_transb(hidden_states, 0., att_scores, v, 1.0);
        
        }
    }

}

// let mut score: Tensor<f32> = Tensor::default(&vec![seq_len, total_seq_len]);
/// 计算过程如下：
/// hidden = rms_norm(residual)
/// gate = hidden @ gate_weight.T
/// up = hidden @ up_weight.T
/// hidden = gate * sigmoid(gate) * up ## silu
/// hidden = hidden @ down_weight.T
/// residual = hidden + residual
fn mlp(
    residual: &mut Tensor<f32>,
    hidden_states: &mut Tensor<f32>,
    gate: &mut Tensor<f32>,
    up: &mut Tensor<f32>,
    w_up: &Tensor<f32>,
    w_down: &Tensor<f32>,
    w_gate: &Tensor<f32>,
    rms_w: &Tensor<f32>,
    eps: f32,
) {
    rms_norm(hidden_states, residual, rms_w, eps);
    matmul_transb(gate, 0., &hidden_states, w_gate, 1.);
    matmul_transb(up, 0., &hidden_states, w_up, 1.);
    silu(up, &gate);
    matmul_transb(hidden_states, 0., up, w_down, 1.);
    let residual_ = matadd(&hidden_states, &residual);
    copy_mat(residual, &residual_);
}

#[test]
pub fn test_mlp() {
    let seq_len = 4;
    let d = 2;
    let di = 3;
    let mut residual = Tensor::<f32>::new(vec![1., 1., 1., 1., 1., 1., 1., 1.], &vec![seq_len, d]);
    let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, d]);
    let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let mut up_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let w_up = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let w_down = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![d, di]);
    let w_gate = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let rms_w = Tensor::<f32>::new(vec![1., 1.], &vec![d]);
    let eps = 1e-6;
    mlp(
        &mut residual,
        &mut hidden_states,
        &mut gate_buf,
        &mut up_buf,
        &w_up,
        &w_down,
        &w_gate,
        &rms_w,
        eps,
    );

    assert!(residual.close_to(
        &Tensor::<f32>::new(
            vec![
                1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964,
                1.7290739
            ],
            &vec![seq_len, d]
        ),
        1e-3
    ))
}

#[test]
pub fn test_load_safetensors() {
    use std::path::PathBuf;
    use crate::tensor::float_eq;
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let model = Llama::from_safetensors(model_dir);
    assert_eq!(model.vocab, 2048);
    assert_eq!(model.n_layers, 2);
    assert_eq!(model.n_q_h, 8);
    assert_eq!(model.n_kv_h, 4);
    assert_eq!(model.d, 128);
    assert_eq!(model.dqkv, 16);
    assert_eq!(model.di, 384);

    assert!(float_eq(&model.params.embedding_table.data()[50], &0.14453125, 1e-6));
    assert_eq!(model.params.lm_head.data()[10], model.params.embedding_table.data()[10]);
    assert!(float_eq(&model.params.rms_att_w[0].data()[10], &0.18652344, 1e-6));
    assert!(float_eq(&model.params.rms_ffn_w[1].data()[10], &0.32421875, 1e-6));
    assert!(float_eq(&model.params.rms_out_w.data()[100], &0.73046875, 1e-6));
    assert!(float_eq(&model.params.w_down[0].data()[100], &-0.0625, 1e-6));
    assert!(float_eq(&model.params.w_up[0].data()[100], &1.46875, 1e-6));
    assert!(float_eq(&model.params.w_gate[1].data()[100], &0.296875, 1e-6));
    assert!(float_eq(&model.params.wq[1].data()[100], &0.032226563, 1e-6));
    assert!(float_eq(&model.params.wk[1].data()[100], &-0.21386719, 1e-6));
    assert!(float_eq(&model.params.wv[0].data()[100], &0.041015625, 1e-6));
    assert!(float_eq(&model.params.wo[0].data()[100], &0.01965332, 1e-6));

}

#[test]
fn safetensors_print(){
    // 打开 .safetensors 文件
    let mut file = File::open("models/story/model.safetensors").expect("Unable to open file");

    // 读取文件内容
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).expect("Unable to read file");

    // 解析 SafeTensors
    let safetensors = SafeTensors::deserialize(&buffer).expect("Unable to deserialize safetensors");


    let tensor = safetensors.tensor("lm_head.weight").expect("Unable to get tensor");

    // let tensor = {
    //     let p:usize=tensor.shape().iter().product();
    //     // 获取引用，只目前只转换成f32类型
    //    let new_data=unsafe { slice::from_raw_parts(tensor.data().as_ptr() as *const f32, p)};
    //    // 生成新对象
    //     Tensor::new(Vec::from(new_data), &tensor.shape().to_vec())
    // };

    

    // println!("50: {}", tensor.data()[50]);

    // 打印 SafeTensors 的结构
    for (name, tensor) in safetensors.tensors() {
        println!("Tensor name: {}", name);
        println!("Shape: {:?}", tensor.shape());
        println!("Data type: {:?}", tensor.dtype());
        println!("-------------------------");
    }
}
