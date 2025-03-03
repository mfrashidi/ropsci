export interface Task {
    id: string
    title: string
    subtitle?: string
    status: "pending" | "loading" | "completed"
  }
  
export interface OnboardingProps {
    tasks: Task[]
    currentTaskId: string
    onComplete?: () => void
}
  
  