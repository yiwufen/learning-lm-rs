use crate::tensor::Tensor;

/// get (row) vectors from a 2D table given a list of indices
pub fn gather(y: &mut Tensor<f32>, indices: &Tensor<u32>, table: &Tensor<f32>) {
    let length = indices.size();
    let table_shape = table.shape();
    assert!(table_shape.len() == 2);
    let dim = table_shape[1];
    assert!(y.size() == length * dim);
    for i in 0..length {
        let src = &table.data()[indices.data()[i] as usize * dim..][..dim];
        let dst = &mut unsafe { y.data_mut() }[i * dim..][..dim];
        dst.copy_from_slice(src);
    }
}

/// RoPE: Rotary Positional Embedding
pub fn rope(y: &mut Tensor<f32>, start_pos: usize, theta: f32) {
    let shape = y.shape();
    assert!(shape.len() == 3);
    let seq_len = shape[0];
    let n_heads = shape[1];
    let d = shape[2];
    let data = unsafe { y.data_mut() };
    for tok in 0..seq_len {
        let pos = start_pos + tok;
        for head in 0..n_heads {
            for i in 0..d / 2 {
                let a = data[tok * n_heads * d + head * d + i];
                let b = data[tok * n_heads * d + head * d + i + d / 2];
                let freq = pos as f32 / theta.powf((i * 2) as f32 / d as f32);
                let (sin, cos) = freq.sin_cos();
                data[tok * n_heads * d + head * d + i] = a * cos - b * sin;
                data[tok * n_heads * d + head * d + i + d / 2] = b * cos + a * sin;
            }
        }
    }
}

/// softmax(x) = exp(x - max) / sum(exp(x - max))
/// 
/// y = softmax(mask(x))
pub fn masked_softmax(y: &mut Tensor<f32>) {
    let ndim = y.shape().len();
    assert!(ndim >= 2);
    let seq_len = y.shape()[ndim - 2];
    let total_seq_len = y.shape()[ndim - 1];
    let batch = y.size() / (seq_len * total_seq_len);
    let data = unsafe { y.data_mut() };
    for b in 0..batch {
        let base = b * seq_len * total_seq_len;
        for i in 0..seq_len {
            let offset = base + i * total_seq_len;
            let boundary = total_seq_len - seq_len + i + 1;

            let max = data[offset..offset + boundary]
                .iter()
                .fold(data[offset], |a, b| a.max(*b));

            let sum = (0..boundary)
                .map(|j| {
                    let e = (data[offset + j] - max).exp();
                    data[offset + j] = e;
                    e
                })
                .sum::<f32>();

            (0..boundary).for_each(|j| data[offset + j] /= sum);
            (boundary..total_seq_len).for_each(|j| data[offset + j] = 0.0);
        }
    }
}

pub fn rms_norm(y: &mut Tensor<f32>, x: &Tensor<f32>, w: &Tensor<f32>, epsilon: f32) {
    let (m,n) = (y.shape()[0], y.shape()[1]);

    assert!(m == x.shape()[0] && n == x.shape()[1]);
    assert!(n == w.size());

    let _y = unsafe { y.data_mut() };
    let _x = x.data();
    let _w = w.data();


    // Step 3: Normalize and store result
    for i in 0..m {
        let mut sum_x_i = 0.0;
        for j in 0..n {
            sum_x_i +=  _x[i * n + j] * _x[i * n + j];
        }
        let rms = (sum_x_i / n as f32 + epsilon).sqrt();
        for j in 0..n {
            _y[i * n + j] = _x[i * n + j] * _w[j] / rms ;
        }
    }
}

// y = sigmoid(x) * x * y
// hint: this is an element-wise operation
pub fn silu(y: &mut Tensor<f32>, x: &Tensor<f32>) {
    let len = y.size();
    assert!(len == x.size());

    let _y = unsafe { y.data_mut() };
    let _x = x.data();
    
    for i in 0..len {
        let x = _x[i];
        _y[i] *= x / (1. + (-x).exp());
    }
    
}

// C = beta * C + alpha * A @ B^T
// hint: You don't need to do an explicit transpose of B
pub fn matmul_transb(c: &mut Tensor<f32>, beta: f32, a: &Tensor<f32>, b: &Tensor<f32>, alpha: f32) {
    let c_ = matmul(a, &transpose(b));
    let c_ = f32_mul_mat(alpha, &c_);
    let c_ = matadd(&f32_mul_mat(beta, c), &c_);
    copy_mat(c, &c_);
}

/// Copy the data from a to c
pub fn copy_mat(c: &mut Tensor<f32>, a: &Tensor<f32>) {
    let c_shape = c.shape();
    let a_shape = a.shape();
    assert!(c_shape == a_shape);
    let m = c_shape[0];
    let n = c_shape[1];
    for i in 0..m {
        for j in 0..n {
            unsafe {
                c.data_mut()[i * n + j] = a.data()[i * n + j];
            }
        }
    }
}

pub fn f32_mul_mat(beta: f32, a: &Tensor<f32>) -> Tensor<f32> {
    let a_shape = a.shape();
    let m = a_shape[0];
    let n = a_shape[1];
    let mut c = Tensor::<f32>::new(vec![0.0; m * n], &vec![m, n]);
    for i in 0..m {
        for j in 0..n {
            unsafe {
                c.data_mut()[i * n + j] = beta * a.data()[i * n + j];
            }
        }
    }
    c
}

/// Element-wise addition of two tensors of the same shape
/// 返回 A + B
pub fn matadd(a: &Tensor<f32>, b: &Tensor<f32>) -> Tensor<f32> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    assert!(a_shape == b_shape);
    let m = a_shape[0];
    let n = a_shape[1];
    let mut c = Tensor::<f32>::new(vec![0.0; m * n], &vec![m, n]);
    for i in 0..m {
        for j in 0..n {
            unsafe {
                c.data_mut()[i * n + j] = a.data()[i * n + j] + b.data()[i * n + j];
            }
        }
    }
    c
}

/// Matrix multiplication of two tensors
/// 返回 A * B
pub fn matmul(x: &Tensor<f32>, y: &Tensor<f32>) -> Tensor<f32> {
    let x_shape = x.shape();
    let y_shape = y.shape();
    assert!(x_shape.len() == 2);
    assert!(y_shape.len() == 2);
    assert!(x_shape[1] == y_shape[0]);
    let m = x_shape[0];
    let n = x_shape[1];
    let k = y_shape[1];
    let mut z = Tensor::<f32>::new(vec![0.0; m * k], &vec![m, k]);
    for i in 0..m {
        for j in 0..k {
            for l in 0..n {
                unsafe {
                    z.data_mut()[i * k + j] += x.data()[i * n + l] * y.data()[l * k + j];
                }
            }
        }
    }
    z
}

/// Transpose a 2D tensor  (m, n) -> (n, m) 转置矩阵
/// 返回 x 的转置
pub fn transpose(x: &Tensor<f32>) -> Tensor<f32> {
    let x_shape = x.shape();
    assert!(x_shape.len() == 2);
    let m = x_shape[0];
    let n = x_shape[1];
    let mut y = Tensor::<f32>::new(vec![0.0; m * n], &vec![n, m]);
    for i in 0..m {
        for j in 0..n {
            unsafe {
                y.data_mut()[j * m + i] = x.data()[i * n + j];
            }
        }
    }
    y
}

/// Dot product of two tensors (treated as vectors)
/// 返回 x 和 y 的点积
#[allow(unused)]
pub fn dot(x: &Tensor<f32>, y: &Tensor<f32>) -> f32 {
    let len = x.size();
    assert!(len == y.size());
    let x_ = x.data();
    let y_ = y.data();
    let mut sum = 0.0;
    for i in 0..len {
        sum += x_[i] * y_[i];
    }
    sum
}

// Sample a index from a tensor (treated as a probability vector)
pub fn random_sample(x: &Tensor<f32>, top_p: f32, top_k: u32, temperature: f32) -> u32 {
    assert!(x.shape()[x.shape().len() - 1] == x.size());
    if temperature <= 0. || top_k < 2 || top_p <= 0. {
        return x
            .data()
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0 as _;
    }

    #[derive(Clone, Copy, PartialEq, Debug)]
    struct Probability {
        val: f32,
        tok: u32,
    }
    impl Eq for Probability {}
    impl PartialOrd for Probability {
        #[inline]
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Probability {
        #[inline]
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            match self.val.total_cmp(&other.val) {
                std::cmp::Ordering::Equal => self.tok.cmp(&other.tok),
                ord => ord.reverse(),
            }
        }
    }
    impl From<(usize, &f32)> for Probability {
        #[inline]
        fn from((i, p): (usize, &f32)) -> Self {
            Self {
                val: p.clone(),
                tok: i as _,
            }
        }
    }

    // sort
    let mut logits = x
        .data()
        .iter()
        .enumerate()
        .map(Probability::from)
        .collect::<Vec<_>>();
    logits.sort_unstable();
    let max = core::mem::replace(&mut logits[0].val, 1.);
    // softmax & sum
    for i in 1..logits.len() {
        logits[i].val = logits[i - 1].val + ((logits[i].val - max) / temperature).exp();
    }
    // topk & topp & random
    let pk = logits[(top_k as usize).min(logits.len()) - 1].val;
    let pp = logits[logits.len() - 1].val * top_p;
    let plimit = rand::random::<f32>() * f32::min(pk, pp);
    // sample
    logits.iter().find(|p| p.val >= plimit).unwrap().tok
}

// Your implementation should at least pass the following tests:
#[test]
fn test_silu() {
    let mut y = Tensor::<f32>::new(vec![2., 3., 4.], &vec![1, 3]);
    let x = Tensor::<f32>::new(vec![1., 2., 3.], &vec![1, 3]);
    silu(&mut y, &x);
    assert!(y.close_to(
        &Tensor::<f32>::new(vec![1.4621172, 5.2847824, 11.43089], &vec![1, 3]),
        1e-3
    ));
}

#[test]
fn test_rms_norm() {
    let mut y = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let x = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let w = Tensor::<f32>::new(vec![1., 2.], &vec![2]);
    rms_norm(&mut y, &x, &w, 1e-6);
    assert!(y.close_to(
        &Tensor::<f32>::new(
            vec![0.6324554, 2.5298216, 0.8485281, 2.2627416],
            &vec![2, 2]
        ),

        
        1e-3
    ));
}

#[test]
fn test_matmul_transb() {
    let mut c = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let a = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    let b = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    matmul_transb(&mut c, 1., &a, &b, 1.);
    c.print();
    assert!(c.close_to(
        &Tensor::<f32>::new(vec![15., 34., 35., 81.], &vec![2, 2]),
        1e-3
    ));
}
