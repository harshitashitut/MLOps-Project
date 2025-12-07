import * as React from "react";

const Button = React.forwardRef(({ className = "", variant = "default", size = "default", ...props }, ref) => {
  const baseStyles = "inline-flex items-center justify-center rounded-md font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 disabled:pointer-events-none disabled:opacity-50";
  
  const variants = {
    default: "border border-gray-300 bg-white hover:bg-gray-50",
    outline: "bg-purple-600 text-white hover:bg-purple-700",
  };
  
  const sizes = {
    default: "h-10 px-4 py-2",
    lg: "h-11 px-8",
  };

  return (
    <button
      className={`${baseStyles} ${variants[variant]} ${sizes[size]} ${className}`}
      ref={ref}
      {...props}
    />
  );
});

Button.displayName = "Button";

export { Button };