//! Replay pywebnn's recorded operations into RustNN's backend-free builder.
//! This adapter keeps serialization in RustNN, including external-weight handling.
use rustnn::mlcontext::{MLGraphBuilder, MLOperand};
use rustnn::Operation;

fn operand(id: u32) -> MLOperand {
    MLOperand::from(id)
}

pub(super) fn record_operation(
    builder: &mut MLGraphBuilder<'static, 'static>,
    operation: &Operation,
) -> rustnn::mlgraphbuilder::Result<Vec<MLOperand>> {
    let result = match operation {
        Operation::Add { a, b, options, .. } => vec![builder.add_with_options(
            operand(*a),
            operand(*b),
            options.clone().unwrap_or_default(),
        )?],
        Operation::Sub { a, b, options, .. } => vec![builder.sub_with_options(
            operand(*a),
            operand(*b),
            options.clone().unwrap_or_default(),
        )?],
        Operation::Mul { a, b, options, .. } => vec![builder.mul_with_options(
            operand(*a),
            operand(*b),
            options.clone().unwrap_or_default(),
        )?],
        Operation::Div { a, b, options, .. } => vec![builder.div_with_options(
            operand(*a),
            operand(*b),
            options.clone().unwrap_or_default(),
        )?],
        Operation::Pow { a, b, options, .. } => vec![builder.pow_with_options(
            operand(*a),
            operand(*b),
            options.clone().unwrap_or_default(),
        )?],
        Operation::Max { a, b, options, .. } => vec![builder.max_with_options(
            operand(*a),
            operand(*b),
            options.clone().unwrap_or_default(),
        )?],
        Operation::Min { a, b, options, .. } => vec![builder.min_with_options(
            operand(*a),
            operand(*b),
            options.clone().unwrap_or_default(),
        )?],
        Operation::Matmul { a, b, options, .. } => vec![builder.matmul_with_options(
            operand(*a),
            operand(*b),
            options.clone().unwrap_or_default(),
        )?],
        Operation::Equal { a, b, options, .. } => vec![builder.equal_with_options(
            operand(*a),
            operand(*b),
            options.clone().unwrap_or_default(),
        )?],
        Operation::Greater { a, b, options, .. } => vec![builder.greater_with_options(
            operand(*a),
            operand(*b),
            options.clone().unwrap_or_default(),
        )?],
        Operation::GreaterOrEqual { a, b, options, .. } => vec![builder
            .greater_or_equal_with_options(
                operand(*a),
                operand(*b),
                options.clone().unwrap_or_default(),
            )?],
        Operation::Lesser { a, b, options, .. } => vec![builder.lesser_with_options(
            operand(*a),
            operand(*b),
            options.clone().unwrap_or_default(),
        )?],
        Operation::LesserOrEqual { a, b, options, .. } => vec![builder
            .lesser_or_equal_with_options(
                operand(*a),
                operand(*b),
                options.clone().unwrap_or_default(),
            )?],
        Operation::NotEqual { a, b, options, .. } => vec![builder.not_equal_with_options(
            operand(*a),
            operand(*b),
            options.clone().unwrap_or_default(),
        )?],
        Operation::Abs { input, options, .. } => {
            vec![builder.abs_with_options(operand(*input), options.clone().unwrap_or_default())?]
        }
        Operation::Ceil { input, options, .. } => {
            vec![builder.ceil_with_options(operand(*input), options.clone().unwrap_or_default())?]
        }
        Operation::Cos { input, options, .. } => {
            vec![builder.cos_with_options(operand(*input), options.clone().unwrap_or_default())?]
        }
        Operation::Exp { input, options, .. } => {
            vec![builder.exp_with_options(operand(*input), options.clone().unwrap_or_default())?]
        }
        Operation::Floor { input, options, .. } => {
            vec![builder.floor_with_options(operand(*input), options.clone().unwrap_or_default())?]
        }
        Operation::Log { input, options, .. } => {
            vec![builder.log_with_options(operand(*input), options.clone().unwrap_or_default())?]
        }
        Operation::Neg { input, options, .. } => {
            vec![builder.neg_with_options(operand(*input), options.clone().unwrap_or_default())?]
        }
        Operation::Relu { input, options, .. } => {
            vec![builder.relu_with_options(operand(*input), options.clone().unwrap_or_default())?]
        }
        Operation::Sigmoid { input, options, .. } => {
            vec![builder
                .sigmoid_with_options(operand(*input), options.clone().unwrap_or_default())?]
        }
        Operation::Sin { input, options, .. } => {
            vec![builder.sin_with_options(operand(*input), options.clone().unwrap_or_default())?]
        }
        Operation::Sqrt { input, options, .. } => {
            vec![builder.sqrt_with_options(operand(*input), options.clone().unwrap_or_default())?]
        }
        Operation::Tan { input, options, .. } => {
            vec![builder.tan_with_options(operand(*input), options.clone().unwrap_or_default())?]
        }
        Operation::Tanh { input, options, .. } => {
            vec![builder.tanh_with_options(operand(*input), options.clone().unwrap_or_default())?]
        }
        Operation::Erf { input, options, .. } => {
            vec![builder.erf_with_options(operand(*input), options.clone().unwrap_or_default())?]
        }
        Operation::Reciprocal { input, options, .. } => vec![builder
            .reciprocal_with_options(operand(*input), options.clone().unwrap_or_default())?],
        Operation::Sign { input, options, .. } => {
            vec![builder.sign_with_options(operand(*input), options.clone().unwrap_or_default())?]
        }
        Operation::LogicalAnd { a, b, options, .. } => vec![builder.logical_and_with_options(
            operand(*a),
            operand(*b),
            options.clone().unwrap_or_default(),
        )?],
        Operation::LogicalOr { a, b, options, .. } => vec![builder.logical_or_with_options(
            operand(*a),
            operand(*b),
            options.clone().unwrap_or_default(),
        )?],
        Operation::LogicalNot { input, options, .. } => vec![builder
            .logical_not_with_options(operand(*input), options.clone().unwrap_or_default())?],
        Operation::LogicalXor { a, b, options, .. } => vec![builder.logical_xor_with_options(
            operand(*a),
            operand(*b),
            options.clone().unwrap_or_default(),
        )?],
        Operation::Where {
            condition,
            true_value,
            false_value,
            options,
            ..
        } => vec![builder.where_with_options(
            operand(*condition),
            operand(*true_value),
            operand(*false_value),
            options.clone().unwrap_or_default(),
        )?],
        Operation::Identity { input, options, .. } => {
            vec![builder
                .identity_with_options(operand(*input), options.clone().unwrap_or_default())?]
        }
        Operation::ArgMin {
            input,
            axis,
            options,
            ..
        } => vec![builder.arg_min_with_options(
            operand(*input),
            *axis,
            options.clone().unwrap_or_default(),
        )?],
        Operation::ArgMax {
            input,
            axis,
            options,
            ..
        } => vec![builder.arg_max_with_options(
            operand(*input),
            *axis,
            options.clone().unwrap_or_default(),
        )?],
        Operation::BatchNormalization {
            input,
            mean,
            variance,
            options,
            ..
        } => vec![builder.batch_normalization_with_options(
            operand(*input),
            operand(*mean),
            operand(*variance),
            options.clone().unwrap_or_default(),
        )?],
        Operation::Cast {
            input,
            data_type,
            options,
            ..
        } => vec![builder.cast_with_options(
            operand(*input),
            *data_type,
            options.clone().unwrap_or_default(),
        )?],
        Operation::Clamp { input, options, .. } => {
            vec![builder.clamp_with_options(operand(*input), options.clone().unwrap_or_default())?]
        }
        Operation::Conv2d {
            input,
            filter,
            options,
            ..
        } => vec![builder.conv2d_with_options(
            operand(*input),
            operand(*filter),
            options.clone().unwrap_or_default(),
        )?],
        Operation::ConvTranspose2d {
            input,
            filter,
            options,
            ..
        } => vec![builder.conv_transpose2d_with_options(
            operand(*input),
            operand(*filter),
            options.clone().unwrap_or_default(),
        )?],
        Operation::Concat {
            inputs,
            axis,
            options,
            ..
        } => vec![builder.concat_with_options(
            &inputs.iter().copied().map(operand).collect::<Vec<_>>(),
            *axis,
            options.clone().unwrap_or_default(),
        )?],
        Operation::CumulativeSum {
            input,
            axis,
            options,
            ..
        } => vec![builder.cumulative_sum_with_options(
            operand(*input),
            *axis,
            options.clone().unwrap_or_default(),
        )?],
        Operation::Expand {
            input,
            new_shape,
            options,
            ..
        } => vec![builder.expand_with_options(
            operand(*input),
            new_shape.clone(),
            options.clone().unwrap_or_default(),
        )?],
        Operation::Elu { input, options, .. } => {
            vec![builder.elu_with_options(operand(*input), options.clone().unwrap_or_default())?]
        }
        Operation::Gather {
            input,
            indices,
            batch_dimensions: _,
            options,
            ..
        } => vec![builder.gather_with_options(
            operand(*input),
            operand(*indices),
            options.clone().unwrap_or_default(),
        )?],
        Operation::GatherElements {
            input,
            indices,
            batch_dimensions: _,
            options,
            ..
        } => vec![builder.gather_elements_with_options(
            operand(*input),
            operand(*indices),
            options.clone().unwrap_or_default(),
        )?],
        Operation::Gemm { a, b, options, .. } => vec![builder.gemm_with_options(
            operand(*a),
            operand(*b),
            options.clone().unwrap_or_default(),
        )?],
        Operation::Gru {
            input,
            weight,
            recurrence,
            steps,
            hidden_size,
            options,
            ..
        } => builder.gru_with_options(
            operand(*input),
            operand(*weight),
            operand(*recurrence),
            *steps,
            *hidden_size,
            options.clone().unwrap_or_default(),
        )?,
        Operation::GruCell {
            input,
            weight,
            recurrence,
            hidden_state,
            hidden_size,
            options,
            ..
        } => vec![builder.gru_cell_with_options(
            operand(*input),
            operand(*weight),
            operand(*recurrence),
            operand(*hidden_state),
            *hidden_size,
            options.clone().unwrap_or_default(),
        )?],
        Operation::HardSigmoid { input, options, .. } => vec![builder
            .hard_sigmoid_with_options(operand(*input), options.clone().unwrap_or_default())?],
        Operation::HardSwish { input, options, .. } => vec![builder
            .hard_swish_with_options(operand(*input), options.clone().unwrap_or_default())?],
        Operation::InstanceNormalization { input, options, .. } => vec![builder
            .instance_normalization_with_options(
                operand(*input),
                options.clone().unwrap_or_default(),
            )?],
        Operation::LayerNormalization { input, options, .. } => vec![builder
            .layer_normalization_with_options(
                operand(*input),
                options.clone().unwrap_or_default(),
            )?],
        Operation::LeakyRelu { input, options, .. } => vec![builder
            .leaky_relu_with_options(operand(*input), options.clone().unwrap_or_default())?],
        Operation::Linear { input, options, .. } => {
            vec![builder
                .linear_with_options(operand(*input), options.clone().unwrap_or_default())?]
        }
        Operation::Lstm {
            input,
            weight,
            recurrence,
            steps,
            hidden_size,
            options,
            ..
        } => builder.lstm_with_options(
            operand(*input),
            operand(*weight),
            operand(*recurrence),
            *steps,
            *hidden_size,
            options.clone().unwrap_or_default(),
        )?,
        Operation::LstmCell {
            input,
            weight,
            recurrence,
            hidden_state,
            cell_state,
            hidden_size,
            options,
            ..
        } => builder.lstm_cell_with_options(
            operand(*input),
            operand(*weight),
            operand(*recurrence),
            operand(*hidden_state),
            operand(*cell_state),
            *hidden_size,
            options.clone().unwrap_or_default(),
        )?,
        Operation::Pad {
            input,
            beginning_padding,
            ending_padding,
            options,
            ..
        } => vec![builder.pad_with_options(
            operand(*input),
            beginning_padding.clone(),
            ending_padding.clone(),
            options.clone().unwrap_or_default(),
        )?],
        Operation::AveragePool2d { input, options, .. } => vec![builder
            .average_pool2d_with_options(operand(*input), options.clone().unwrap_or_default())?],
        Operation::MaxPool2d { input, options, .. } => vec![builder
            .max_pool2d_with_options(operand(*input), options.clone().unwrap_or_default())?],
        Operation::L2Pool2d { input, options, .. } => {
            vec![builder
                .l2_pool2d_with_options(operand(*input), options.clone().unwrap_or_default())?]
        }
        Operation::GlobalAveragePool { input, options, .. } => vec![builder
            .global_average_pool_with_options(
                operand(*input),
                options.clone().unwrap_or_default(),
            )?],
        Operation::GlobalMaxPool { input, options, .. } => vec![builder
            .global_max_pool_with_options(operand(*input), options.clone().unwrap_or_default())?],
        Operation::ReduceSum { input, options, .. } => vec![builder
            .reduce_sum_with_options(operand(*input), options.clone().unwrap_or_default())?],
        Operation::ReduceMean { input, options, .. } => vec![builder
            .reduce_mean_with_options(operand(*input), options.clone().unwrap_or_default())?],
        Operation::ReduceMax { input, options, .. } => vec![builder
            .reduce_max_with_options(operand(*input), options.clone().unwrap_or_default())?],
        Operation::ReduceMin { input, options, .. } => vec![builder
            .reduce_min_with_options(operand(*input), options.clone().unwrap_or_default())?],
        Operation::ReduceProduct { input, options, .. } => vec![builder
            .reduce_product_with_options(operand(*input), options.clone().unwrap_or_default())?],
        Operation::ReduceL1 { input, options, .. } => {
            vec![builder
                .reduce_l1_with_options(operand(*input), options.clone().unwrap_or_default())?]
        }
        Operation::ReduceL2 { input, options, .. } => {
            vec![builder
                .reduce_l2_with_options(operand(*input), options.clone().unwrap_or_default())?]
        }
        Operation::ReduceLogSum { input, options, .. } => vec![builder
            .reduce_log_sum_with_options(operand(*input), options.clone().unwrap_or_default())?],
        Operation::ReduceLogSumExp { input, options, .. } => vec![builder
            .reduce_log_sum_exp_with_options(
                operand(*input),
                options.clone().unwrap_or_default(),
            )?],
        Operation::ReduceSumSquare { input, options, .. } => vec![builder
            .reduce_sum_square_with_options(
                operand(*input),
                options.clone().unwrap_or_default(),
            )?],
        Operation::Reshape {
            input,
            new_shape,
            options,
            ..
        } => vec![builder.reshape_with_options(
            operand(*input),
            new_shape.clone(),
            options.clone().unwrap_or_default(),
        )?],
        Operation::Resample2d { input, options, .. } => vec![builder
            .resample2d_with_options(operand(*input), options.clone().unwrap_or_default())?],
        Operation::Reverse { input, options, .. } => {
            vec![builder
                .reverse_with_options(operand(*input), options.clone().unwrap_or_default())?]
        }
        Operation::ScatterElements {
            input,
            indices,
            updates,
            options,
            ..
        } => vec![builder.scatter_elements_with_options(
            operand(*input),
            operand(*indices),
            operand(*updates),
            options.clone().unwrap_or_default(),
        )?],
        Operation::Softmax {
            input,
            axis,
            options,
            ..
        } => vec![builder.softmax_with_options(
            operand(*input),
            *axis,
            options.clone().unwrap_or_default(),
        )?],
        Operation::Slice {
            input,
            starts,
            sizes,
            options,
            ..
        } => vec![builder.slice_with_options(
            operand(*input),
            starts,
            sizes,
            options.clone().unwrap_or_default(),
        )?],
        Operation::Split {
            input,
            splits,
            split_equal_parts,
            options,
            ..
        } => {
            if let Some(count) = split_equal_parts {
                builder.split_equal_with_options(
                    operand(*input),
                    *count,
                    options.clone().unwrap_or_default(),
                )?
            } else {
                builder.split_with_options(
                    operand(*input),
                    splits,
                    options.clone().unwrap_or_default(),
                )?
            }
        }
        Operation::Transpose { input, options, .. } => {
            vec![builder
                .transpose_with_options(operand(*input), options.clone().unwrap_or_default())?]
        }
        Operation::Squeeze { input, options, .. } => {
            vec![builder
                .squeeze_with_options(operand(*input), options.clone().unwrap_or_default())?]
        }
        Operation::Unsqueeze { input, options, .. } => {
            vec![builder
                .unsqueeze_with_options(operand(*input), options.clone().unwrap_or_default())?]
        }
        Operation::Tile {
            input,
            repetitions,
            options,
            ..
        } => vec![builder.tile_with_options(
            operand(*input),
            repetitions.clone(),
            options.clone().unwrap_or_default(),
        )?],
        Operation::Triangular { input, options, .. } => vec![builder
            .triangular_with_options(operand(*input), options.clone().unwrap_or_default())?],
        Operation::Prelu {
            input,
            slope,
            options,
            ..
        } => vec![builder.prelu_with_options(
            operand(*input),
            operand(*slope),
            options.clone().unwrap_or_default(),
        )?],
        Operation::QuantizeLinear {
            input,
            scale,
            zero_point,
            options,
            ..
        } => vec![builder.quantize_linear_with_options(
            operand(*input),
            operand(*scale),
            zero_point.map(operand),
            options.clone().unwrap_or_default(),
        )?],
        Operation::DequantizeLinear {
            input,
            scale,
            zero_point,
            options,
            ..
        } => vec![builder.dequantize_linear_with_options(
            operand(*input),
            operand(*scale),
            zero_point.map(operand),
            options.clone().unwrap_or_default(),
        )?],
        Operation::Softplus { input, options, .. } => {
            vec![builder
                .softplus_with_options(operand(*input), options.clone().unwrap_or_default())?]
        }
        Operation::Softsign { input, options, .. } => {
            vec![builder
                .softsign_with_options(operand(*input), options.clone().unwrap_or_default())?]
        }
        Operation::Gelu { input, options, .. } => {
            vec![builder.gelu_with_options(operand(*input), options.clone().unwrap_or_default())?]
        }
        Operation::Shape { input, options, .. } => {
            vec![builder.shape_with_options(operand(*input), options.clone().unwrap_or_default())?]
        }
        Operation::ScatterND {
            input,
            indices,
            updates,
            options,
            ..
        } => vec![builder.scatter_nd_with_options(
            operand(*input),
            operand(*indices),
            operand(*updates),
            options.clone().unwrap_or_default(),
        )?],
        Operation::GatherND {
            input,
            indices,
            options,
            ..
        } => vec![builder.gather_nd_with_options(
            operand(*input),
            operand(*indices),
            options.clone().unwrap_or_default(),
        )?],
        Operation::IsNaN { input, options, .. } => {
            vec![builder
                .is_nan_with_options(operand(*input), options.clone().unwrap_or_default())?]
        }
        Operation::IsInfinite { input, options, .. } => vec![builder
            .is_infinite_with_options(operand(*input), options.clone().unwrap_or_default())?],
        Operation::RoundEven { input, options, .. } => vec![builder
            .round_even_with_options(operand(*input), options.clone().unwrap_or_default())?],
    };
    Ok(result)
}
