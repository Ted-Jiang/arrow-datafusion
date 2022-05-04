// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Defines physical expressions that can evaluated at runtime during query execution

use std::any::Any;

use std::fmt::Debug;
use std::ops::BitOrAssign;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, BinaryArray};
use arrow::datatypes::{DataType, Field};
use croaring::treemap::JvmSerializer;
use croaring::Treemap;
use datafusion_common::{DataFusionError, Result, ScalarValue};
use datafusion_expr::Accumulator;

use crate::{AggregateExpr, PhysicalExpr};

#[derive(Debug)]
pub struct KylinBitMapDistinct {
    name: String,
    input_data_type: DataType,
    expr: Arc<dyn PhysicalExpr>,
}

impl KylinBitMapDistinct {
    /// Create a new bitmapDistinct aggregate function.
    pub fn new(
        expr: Arc<dyn PhysicalExpr>,
        name: impl Into<String>,
        input_data_type: DataType,
    ) -> Self {
        Self {
            name: name.into(),
            input_data_type,
            expr,
        }
    }
}

impl AggregateExpr for KylinBitMapDistinct {
    /// Return a reference to Any that can be used for downcasting
    fn as_any(&self) -> &dyn Any {
        self
    }

    /// the field of the final result of this aggregation.
    fn field(&self) -> Result<Field> {
        Ok(Field::new(&self.name, DataType::UInt64, false))
    }

    fn create_accumulator(&self) -> Result<Box<dyn Accumulator>> {
        let accumulator: Box<dyn Accumulator> = match &self.input_data_type {
            DataType::Binary => Box::new(BitmapDistinctCountAccumulator::try_new()),
            other => {
                return Err(DataFusionError::NotImplemented(format!(
                    "Support for 'kylin_bitmap_distinct' for data type {} is not implemented",
                    other
                )));
            }
        };
        Ok(accumulator)
    }

    fn state_fields(&self) -> Result<Vec<Field>> {
        Ok(vec![Field::new(
            &format!("{} {}", self.name.as_str(), "kylin_bitmap_registers"),
            DataType::Binary,
            false,
        )])
    }

    fn expressions(&self) -> Vec<Arc<dyn PhysicalExpr>> {
        vec![self.expr.clone()]
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[derive(Debug)]
struct BitmapDistinctCountAccumulator {
    bitmap: croaring::treemap::Treemap,
}

impl BitmapDistinctCountAccumulator {
    fn try_new() -> Self {
        Self {
            bitmap: croaring::treemap::Treemap::create(),
        }
    }
}

impl Accumulator for BitmapDistinctCountAccumulator {
    //state() can be used by physical nodes to aggregate states together and send them over the network/threads, to combine values.
    fn state(&self) -> Result<Vec<ScalarValue>> {
        //maybe run optimized
        let buffer = self
            .bitmap
            .serialize()
            .expect("fail to serialize in BitmapDistinctCountAccumulator");
        Ok(vec![ScalarValue::Binary(Some(buffer))])
    }

    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let value = &values[0];
        if value.is_empty() {
            return Ok(());
        }
        match value.data_type() {
            DataType::Binary => {
                let array = value.as_any().downcast_ref::<BinaryArray>().unwrap();
                //Do not use self.bitmap = xxx, because '=' has been wrote for 'BitAnd' !.
                // self.bitmap.bitor_assign(Bitmap::fast_or(bitmaps));
                for bytes in array {
                    let bitmap = Treemap::deserialize(bytes.expect("Impossibly got empty binary array from states in kylin_bitmap_distinct")).expect("fail to deserialize");
                    self.bitmap.bitor_assign(bitmap);
                }
            }
            e => {
                return Err(DataFusionError::Internal(format!(
                    "BITMAP_COUNT_DISTINCT is not expected to receive the type {:?}",
                    e
                )));
            }
        }
        Ok(())
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        if states.len() != 1 {
            "states must be 1 in KylinBitMapDistinct".to_string();
        }
        let binary_array = states[0].as_any().downcast_ref::<BinaryArray>().unwrap();
        for array in binary_array {
            let treemap = Treemap::deserialize(array.expect(
                "Impossibly got empty binary array from states in kylin_bitmap_distinct",
            ))
            .expect("fail to deserialize");
            self.bitmap.bitor_assign(treemap);
        }
        Ok(())
    }

    fn evaluate(&self) -> Result<ScalarValue> {
        Ok(ScalarValue::from(self.bitmap.cardinality()))
    }
}
