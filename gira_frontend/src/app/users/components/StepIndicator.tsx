interface StepIndicatorProps {
  currentStep: number;
}

export default function StepIndicator({ currentStep }: StepIndicatorProps) {
  return (
    <div className="mb-8">
      <div className="flex justify-between items-center">
        <div className="flex items-center space-x-4">
          <div className="flex items-center">
            <span 
              className={`text-sm font-medium ${
                currentStep === 1 ? 'text-blue-600' : 'text-gray-500'
              }`}
            >
              Personal Information
            </span>
            <div 
              className={`ml-4 h-0.5 w-20 ${
                currentStep === 1 ? 'bg-blue-600' : 'bg-gray-300'
              }`}
            />
          </div>
          
          <div className="flex items-center">
            <span 
              className={`text-sm font-medium ${
                currentStep === 2 ? 'text-blue-600' : 'text-gray-500'
              }`}
            >
              Additional Information
            </span>
            <div 
              className={`ml-4 h-0.5 w-20 ${
                currentStep === 2 ? 'bg-blue-600' : 'bg-gray-300'
              }`}
            />
          </div>
        </div>
      </div>
    </div>
  );
}