import { useState, useCallback } from "react"
import type { Task } from "../types/onboarding"

export function useOnboarding(initialTasks: Task[]) {
  const [tasks, setTasks] = useState<Task[]>(initialTasks)
  const [currentTaskId, setCurrentTaskId] = useState<string>(initialTasks[0]?.id)

  const completeTask = useCallback(
    (taskId: string) => {
      setTasks((prevTasks) =>
        prevTasks.map((task) => {
          if (task.id === taskId) {
            return { ...task, status: "completed" }
          }
          return task
        }),
      )

      const currentIndex = tasks.findIndex((task) => task.id === taskId)
      const nextTask = tasks[currentIndex + 1]

      if (nextTask) {
        setCurrentTaskId(nextTask.id)
        setTasks((prevTasks) =>
          prevTasks.map((task) => {
            if (task.id === nextTask.id) {
              return { ...task, status: "loading" }
            }
            return task
          }),
        )
      }
    },
    [tasks],
  )

  const startTask = useCallback((taskId: string) => {
    setTasks((prevTasks) =>
      prevTasks.map((task) => {
        if (task.id === taskId) {
          return { ...task, status: "loading" }
        }
        return task
      }),
    )
  }, [])

  return {
    tasks,
    currentTaskId,
    completeTask,
    startTask,
  }
}

