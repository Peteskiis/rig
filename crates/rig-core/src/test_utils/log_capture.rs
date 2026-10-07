//! Captures formatted `tracing` output on the current thread for log assertions.

use std::io::{self, Write};
use std::sync::{Arc, Mutex};

use tracing::Level;
use tracing::subscriber::DefaultGuard;

/// Buffers every event at or above `level` until dropped.
pub(crate) struct CapturedLogs {
    buffer: Arc<Mutex<Vec<u8>>>,
    _guard: DefaultGuard,
}

impl CapturedLogs {
    pub(crate) fn at(level: Level) -> Self {
        let buffer = Arc::new(Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(level)
            .with_ansi(false)
            .without_time()
            .with_writer({
                let buffer = buffer.clone();
                move || SharedWriter(buffer.clone())
            })
            .finish();
        Self {
            buffer,
            _guard: tracing::subscriber::set_default(subscriber),
        }
    }

    pub(crate) fn text(&self) -> String {
        let bytes = self
            .buffer
            .lock()
            .expect("log buffer mutex should not be poisoned")
            .clone();
        String::from_utf8(bytes).expect("captured logs should be valid UTF-8")
    }
}

#[derive(Clone)]
struct SharedWriter(Arc<Mutex<Vec<u8>>>);

impl Write for SharedWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0
            .lock()
            .expect("log buffer mutex should not be poisoned")
            .extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
