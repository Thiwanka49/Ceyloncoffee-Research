import React from 'react';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  size?: 'sm' | 'md' | 'lg';
}

export const Button: React.FC<ButtonProps> = ({ 
  size = 'md', 
  className, 
  children, 
  ...props 
}) => {
  const sizeClasses = {
    sm: 'px-3 py-1 text-sm',
    md: 'px-4 py-2 text-base',
    lg: 'px-6 py-3 text-lg'
  };

  return (
    <button
      className={`
        bg-[#BB956A] hover:bg-[#A07054] text-white 
        rounded-lg transition-colors duration-200
        ${sizeClasses[size]} ${className || ''}
      `}
      {...props}
    >
      {children}
    </button>
  );
};