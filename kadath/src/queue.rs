use std::collections::VecDeque;
use std::sync::Mutex;
use tokio::sync::Notify;

pub struct Queue<T> {
    queue: Mutex<VecDeque<T>>,
    notify: Notify,
    capacity: usize,
}

impl<T> Queue<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            queue: Mutex::new(VecDeque::with_capacity(capacity)),
            notify: Notify::new(),
            capacity,
        }
    }

    pub fn push(&self, value: T) {
        let mut queue = self.queue.lock().unwrap();

        if queue.len() >= self.capacity {
            queue.pop_front();
        }

        queue.push_back(value);

        drop(queue);

        self.notify.notify_one();
    }

    pub async fn pop(&self) -> T {
        loop {
            if let Some(value) = self.queue.lock().unwrap().pop_front() {
                return value;
            }

            self.notify.notified().await;
        }
    }
}